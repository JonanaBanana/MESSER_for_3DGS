import numpy as np
import open3d as o3d
import csv
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2 as cv


######################### CONSTANTS ###############################
voxel_size = 0.1
min_x = 1 #min distance to keep points
max_x = 200 #max distance to keep points
f = 1108.5125019853992
h = 720
w = 1280
px = 640
py = 360
fov_x = 2*np.arctan2(w,(2*f))
fov_y = 2*np.arctan2(h,(2*f))
#projection matrix to project 3d points to image plane
proj_mat = np.array([[f, 0, px, 0],
                    [0, f, py, 0],
                    [0, 0, 1, 0]])
viz = True
#################################################################

########################## PATHS ################################
main_path = '/home/jonathan/Reconstruction/test_stage_warehouse_custom'
pcd_path = os.path.join(main_path,'pcd')
img_path = os.path.join(main_path,'input/')
scans_path = os.path.join(main_path,'scans.pcd')
transform_path = os.path.join(main_path,'transformations.csv')
depth_path = os.path.join(main_path,'depth_images')
depth_json_path = os.path.join(main_path,'depth_params.json')
#################################################################

############# Transformation matrix to camera frame #############
trans_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
#################################################################

#read pointcloud
pcd = o3d.geometry.PointCloud()
pcd_fl = o3d.io.read_point_cloud(scans_path)
pcd_fl.remove_non_finite_points()
pcd_fl = pcd_fl.voxel_down_sample(voxel_size=voxel_size)
pcd_fl,_ = pcd_fl.remove_statistical_outlier(nb_neighbors=10,
                                                std_ratio=1.5)
pcd.points = pcd_fl.points
pcd_down_out = deepcopy(pcd)


mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
if viz == True:
    o3d.visualization.draw_geometries([pcd_down_out,mesh_frame],zoom=0.2,
                                    front=[-1., 0., 0.],
                                    lookat=[0., 0., 0],
                                    up=[0., 0., 1.])


R,_ = np.shape(np.asarray(pcd.points))
point3d_id = np.linspace(0,R-1,R).astype(int)
points = np.asarray(pcd.points)
print("Created 3D point indexing, found ",R,'points!')

#Determine number of images
k = 0
for file in os.listdir(img_path):
    k = k+1    
print('Found ', k,'images!')

#read transform table
with open(transform_path, 'r') as file:
    reader = csv.reader(file)
    transform_data = []
    for row in reader:
        row = [float(i) for i in row]
        transform_data.append(row)
transform_data = np.array(transform_data)
transform = np.reshape(transform_data,(-1,4,4))
N,_,_ = np.shape(transform)
print("Found",N,'transforms!')
print("Processing images...")
for i in range(N):
    print(i,'/',N-1)
    
    if i >= 10:
        im_string = 'img_000' + str(i) + '.jpg'
        if i >= 100:
            im_string = 'img_00' + str(i) + '.jpg'
            if i >= 1000:
                im_string = 'img_0' + str(i) + '.jpg'
                if k >= 10000:
                    im_string = 'img_' + str(i) + '.jpg'
    else:
        im_string = 'img_0000' + str(i) + '.jpg'
        
        
    img = np.asarray(o3d.io.read_image(img_path+im_string))
    np.set_printoptions(suppress=True,precision=3)
    q_transform = np.linalg.inv(transform[i]@trans_mat)
    temp = deepcopy(pcd)
    temp_point3d_id = deepcopy(point3d_id)
    temp.transform(q_transform)

    temp_points = np.asarray(temp.points)
    
    #Initial masking
    x = temp_points[:,0] #3dimensional decomposition of lidar points
    y = temp_points[:,1]
    z = temp_points[:,2]

    #determining angle of each beam in pointcloud both horizontal and vertical angle
    theta_x = np.arctan2(x,z) #2d angles of points correlating to angles of image pixels
    theta_y = np.arctan2(-y,z)

    # filter out points outside the fov angles of the 
    ### Code for filtering points outside the fov of the camera
    positive_mask = np.where((np.abs(theta_x) < (fov_x/2)) & (np.abs(theta_y)<(fov_y/2)) & (z<max_x) & (z > min_x)) #filtering mask
    temp_points = np.squeeze(temp_points[positive_mask,:]) #filtered points
    temp_point3d_id = np.squeeze(temp_point3d_id[positive_mask])

    temp.points = o3d.utility.Vector3dVector(temp_points)
    
    diameter = np.linalg.norm(np.asarray(temp.get_max_bound()) - np.asarray(temp.get_min_bound()))
    camera = [0, 0, 0]
    radius = diameter*2000
    _, positive_mask = temp.hidden_point_removal(camera, radius)
    temp_points = np.squeeze(temp_points[positive_mask])
    temp_point3d_id = np.squeeze(temp_point3d_id[positive_mask])
    ### Code for projecting the point cloud onto the image plane ###
    r,_ = np.shape(temp_points) #shape of filtered array
    extend_homogenous = np.ones((r,1)) #creating homogenous extender (r,1)

    temp_points_homogenous = np.hstack((temp_points,extend_homogenous)) #homogenous array (r,4)

    temp_points_proj = temp_points_homogenous@proj_mat.T #initial projection of points to image plane
    x = np.divide(temp_points_proj[:,0],temp_points_proj[:,2]) #normalizing to fit with image size
    y = np.divide(temp_points_proj[:,1],temp_points_proj[:,2])
    z = extend_homogenous
    temp_points_proj = np.column_stack((x,y,z)) # extending array to 3d to visualize as pointcloud (not necessary)
    positive_mask = np.where((temp_points_proj[:,0] < w*0.99) & (temp_points_proj[:,1] < h*0.99) & (temp_points_proj[:,0] > w*0.01) & (temp_points_proj[:,1] > h*0.01) ) #filtering mask

    temp_points_proj = np.squeeze(temp_points_proj[positive_mask,:])
    temp_points = np.squeeze(temp_points[positive_mask,:])
    temp_point3d_id = np.expand_dims((temp_point3d_id[positive_mask]),axis=1)
    temp_points_proj_idx = np.round(temp_points_proj).astype(int) #rounding to corresponding pixel in image
    
    
    idx_y = temp_points_proj_idx[:,0]
    idx_x = temp_points_proj_idx[:,1]
    depth_val = np.linalg.norm(temp_points,axis=1)
    
    img_depth = img[:,:,0]*0
    M = np.squeeze(np.shape(idx_x))
    for i in range(M):
        if img_depth[idx_x[i],idx_y[i]] == 0:
            img_depth[idx_x[i],idx_y[i]] = depth_val[i]
        else:
            img_depth[idx_x[i],idx_y[i]] = (img_depth[idx_x[i],idx_y[i]] + depth_val[i])*0.5
    
    #img_depth = cv.GaussianBlur(img_depth,(5,5),0)
    
    plt.gray()
    plt.imshow(img_depth)
    plt.show()
    
    depth_data = np.vstack((idx_x,idx_y,depth_val)).T
    #print(depth_data)  

        
