import numpy as np
import open3d as o3d
import csv
import os
from copy import deepcopy
from ament_index_python.packages import get_package_share_directory

######################### CONSTANTS ###############################
use_gt_pose = False
viz = True

fill_background = True
sphere_center = [0,0,0]
sphere_radius = 200 #meters
sphere_num_pts = 50000

hidden_point_removal_factor = 100000

voxel_size = 0.1
min_x = 1 #min distance to keep points
max_x = 400 #max distance to keep points
f = 1108.5125
h = 720
w = 1280
px = 640
py = 360
#################################################################


########################## PATHS ################################
main_path = get_package_share_directory('messer_for_3dgs')
main_path = os.path.join(main_path,'../../captured_data/')
pcd_path = os.path.join(main_path,'pcd')
img_path = os.path.join(main_path,'input/')
accumulated_path = os.path.join(main_path,'pcd/accumulated_point_cloud.pcd')
downsampled_path = os.path.join(main_path,'downsampled_point_cloud.pcd')
scans_path = os.path.join(main_path,'scans.pcd')
transform_path = os.path.join(main_path,'transformations.csv')
output_path = os.path.join(main_path,'point_cloud_color_information.csv')
if use_gt_pose == True:
    transform_path =  os.path.join(main_path,'gt_transformations.csv')
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
pcd_fl,_ = pcd_fl.remove_statistical_outlier(nb_neighbors=10,
                                                std_ratio=2)
pcd_fl = pcd_fl.voxel_down_sample(voxel_size=voxel_size)
pcd.points = pcd_fl.points
if fill_background == True:
    sphere_center,_ = pcd.compute_mean_and_covariance()
    indices = np.arange(0, sphere_num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/sphere_num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    sphere_x = np.cos(theta) * np.sin(phi)*sphere_radius + sphere_center[0]
    sphere_y = np.sin(theta) * np.sin(phi)*sphere_radius + sphere_center[1]
    sphere_z = np.cos(phi)*sphere_radius + sphere_center[2]
    sphere_points = np.vstack((sphere_x,sphere_y,sphere_z)).T
    pcd.points.extend(o3d.utility.Vector3dVector(sphere_points))


print("Saving voxel downsampled point cloud which is used for indexing")
pcd_down_out = deepcopy(pcd)
o3d.io.write_point_cloud(downsampled_path, pcd_down_out, write_ascii=True)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
if viz == True:
    o3d.visualization.draw_geometries([pcd_down_out,mesh_frame],zoom=0.1,
                                    front=[-1., 0., 0.],
                                    lookat=[20., 0., 0],
                                    up=[0., 0., 1.])
    #o3d.visualization.draw_geometries([pcd_down_out,mesh_frame],zoom=0.2,
    #                                    front=[0., 0., -1.],
    #                                    lookat=[0., -2., 20],
    #                                    up=[0., -1., 0.])


fov_x = 2*np.arctan2(w,(2*f))
fov_y = 2*np.arctan2(h,(2*f))
#projection matrix to project 3d points to image plane
proj_mat = np.array([[f, 0, px, 0],
                    [0, f, py, 0],
                    [0, 0, 1, 0]])
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
    camera = [0, 0, 0]
    radius = hidden_point_removal_factor
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
    idx_x = temp_points_proj_idx[:,0]
    idx_y = temp_points_proj_idx[:,1]
    colors_proj = np.array(img[idx_y,idx_x]/255) #using pixel index to determine colors of points
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
        o3d.visualization.draw_geometries([temp,mesh_frame],zoom=0.003,
                                        front=[0., 0., -1.],
                                        lookat=[0., 0., 1],
                                        up=[0., -1., 0.])
        """
        temp_pixel = o3d.geometry.PointCloud()
        temp_pixel.points = o3d.utility.Vector3dVector(temp_points_proj)
        temp_pixel.colors = o3d.utility.Vector3dVector(colors_proj)
        mesh_frame_pixel = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([temp_pixel,mesh_frame_pixel],zoom=100,
                                        front=[0., 0., -1.],
                                        lookat=[0., 360., 640],
                                        up=[0., -1., 0.])"""
    print("Found "+str(M)+" points in image "+str(i))
print("Saving point color information at: "+str(output_path))
np.savetxt(output_path, list_colors, fmt=['%d','%.8f','%.8f','%.8f'], delimiter=",")
        
