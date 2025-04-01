import numpy as np
import open3d as o3d
import csv
import os
from copy import deepcopy


######################### CONSTANTS ###############################
voxel_size = 0.15
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
fast_lio = True
viz = False
#################################################################

########################## PATHS ################################
main_path = '/home/jonathan/Reconstruction/test_stage_chessboard_4'
pcd_path = os.path.join(main_path,'pcd')
img_path = os.path.join(main_path,'input/')
accumulated_path = os.path.join(main_path,'pcd/accumulated_point_cloud.pcd')
downsampled_path = os.path.join(main_path,'downsampled_point_cloud.pcd')
scans_path = os.path.join(main_path,'scans.pcd')
transform_path = os.path.join(main_path,'transformations.csv')
output_path = os.path.join(main_path,'point_cloud_color_information.csv')
#################################################################

############# Transformation matrix to camera frame #############
trans_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
#################################################################

#read pointcloud
if fast_lio == True:
    pcd = o3d.geometry.PointCloud()
    pcd_fl = o3d.io.read_point_cloud(scans_path)
    pcd_fl = pcd_fl.voxel_down_sample(voxel_size=voxel_size)
    pcd_fl,_ = pcd_fl.remove_statistical_outlier(nb_neighbors=10,
                                                    std_ratio=2.0)
    #pcd_fl.transform(np.linalg.inv(trans_mat))
    pcd.points = pcd_fl.points
    print("Saving voxel downsampled point cloud which is used for indexing")
    pcd_down_out = deepcopy(pcd)
    #pcd_down_out.transform(np.linalg.inv(trans_mat))
    o3d.io.write_point_cloud(downsampled_path, pcd_down_out, write_ascii=True)
else:
    pcd = o3d.io.read_point_cloud(accumulated_path)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print("Saving voxel downsampled point cloud which is used for indexing")
    o3d.io.write_point_cloud(downsampled_path, pcd, write_ascii=True)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
if viz == True:
    o3d.visualization.draw_geometries([pcd_down_out,mesh_frame],zoom=0.2,
                                    front=[-1., 0., 0.],
                                    lookat=[20., 0., 0],
                                    up=[0., 0., 1.])
    #o3d.visualization.draw_geometries([pcd_down_out,mesh_frame],zoom=0.2,
    #                                    front=[0., 0., -1.],
    #                                    lookat=[0., -2., 20],
    #                                    up=[0., -1., 0.])


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
    colors = np.ones(np.shape(pcd.points))*0.1
    img = np.asarray(o3d.io.read_image(img_path+'/img_'+str(i)+'.jpg'))
    #q_transform = np.linalg.inv(trans_mat)@np.linalg.pinv(transform[i])
    np.set_printoptions(suppress=True,precision=3)
    #print(transform[i])
    q_transform = np.linalg.inv(transform[i]@trans_mat)
    temp = deepcopy(pcd)
    temp_point3d_id = deepcopy(point3d_id)
    temp.transform(q_transform)
    #if viz == True:
    #    o3d.visualization.draw_geometries([temp,mesh_frame],zoom=0.2,
    #                                    front=[0., 0., -1.],
    #                                    lookat=[0., -2., 20],
    #                                    up=[0., -1., 0.])

    temp_points = np.asarray(temp.points)
    diameter = np.linalg.norm(np.asarray(temp.get_max_bound()) - np.asarray(temp.get_min_bound()))
    camera = [0, 0, 0]
    radius = diameter*600
    _, positive_mask = temp.hidden_point_removal(camera, radius)
    temp_points = np.squeeze(temp_points[positive_mask])
    temp_point3d_id = np.squeeze(temp_point3d_id[positive_mask])
    r_p,_ = np.shape(points)

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
    ### Code for projecting the point cloud onto the image plane ###
    r_new,_ = np.shape(temp_points) #shape of filtered array
    extend_homogenous = np.ones((r_new,1)) #creating homogenous extender (r_new,1)

    temp_points_homogenous = np.hstack((temp_points,extend_homogenous)) #homogenous array (r_new,4)

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
    idx_x = temp_points_proj_idx[:,0]
    idx_y = temp_points_proj_idx[:,1]
    colors_proj = np.array(img[idx_y,idx_x])/255 #using pixel index to determine colors of points
    #print('colors_shape =',np.shape(colors))
    #print('id_shape =',np.shape(temp_point3d_id))
    n = 0
    for idx in temp_point3d_id:
        colors[idx,:] = colors_proj[n,:]
        n = n+1
    M,_ = np.shape(temp_point3d_id)
    if i == 0:
        list_colors = np.hstack((temp_point3d_id,colors_proj))
        #print('list_colors_shape = ',np.shape(list_colors))
    else:
        temp_list = np.hstack((temp_point3d_id,colors_proj))
        list_colors = np.vstack((list_colors,temp_list))
    temp.points = o3d.utility.Vector3dVector(points)
    temp.colors = o3d.utility.Vector3dVector(colors)
    temp.transform(q_transform)    
    if viz == True:
        o3d.visualization.draw_geometries([temp,mesh_frame],zoom=0.2,
                                        front=[0., 0., -1.],
                                        lookat=[0., -2., 20],
                                        up=[0., -1., 0.])
    print("Found "+str(M)+" points in image "+str(i))
print("Saving point color information at: "+str(output_path))
np.savetxt(output_path, list_colors, fmt=['%d','%.8f','%.8f','%.8f'], delimiter=",")
        
