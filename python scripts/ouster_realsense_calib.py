import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
pcd = o3d.io.read_point_cloud("/home/jonathan/ouster_realsense_calib_preprocessed/calib_bag_05.ply")
img = o3d.io.read_image('/home/jonathan/ouster_realsense_calib_preprocessed/calib_bag_05.png')
points = np.asarray(pcd.points)
img = np.asarray(img)

#camera projection matrix 3x3
"""
"camera": {
    "camera_model": "plumb_bob",
    "distortion_coeffs": [
      -0.05607928708195686,
      0.06536319851875305,
      -0.0004867807147093117,
      0.0002132471272489056,
      -0.020375743508338928
    ],
    "intrinsics": [
      641.1700439453125,
      640.310302734375,
      650.11865234375,
      371.19964599609375
    ]
  },
  "results": {
    "T_lidar_camera": [
      0.05134961561749092,
      -0.029777443329282324,
      -0.019582032045242042,
      -0.5082205781632076,
      0.5113221181110232,
      -0.4939747181003944,
      0.48605608044704857
    ],
    "init_T_lidar_camera_auto": [
      0.027960954575839903,
      -0.060802416098065915,
      -0.013959900753576195,
      0.5116573938226513,
      -0.5079262025684697,
      0.49095306029106145,
      -0.48906316226212326
    ]
  }
"""
### Camera information and tra
fx = 641.17 #focal lengths fx and fy, here they are the same 
fy = 640.31
px = 650.12 #principal point offset (center of the image plane relative to sensor corner) in x
py = 371.2 # ppo for y
s = 0 #skew
#height and width in pixels of camera image
h = 720
w = 1280

#The camera 3x4 camera projection matrix
proj_mat = np.array([[fx, s, px, 0],
                     [0, fy, py, 0],
                     [0, 0, 1, 0]])

#Rotation matrix for aligning rotation of Camera and LiDAR
#currently this is an identity matrix since the up-direction of camera and lidar is aligned
rot_mat = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])

#4x4 transformation matrix to transform 3d point from camera frame to LiDAR frame
#This is refined from the calibration result, knowing that the transforms in the simulation environment are much simpler
#To do the inverse transformation simply invert the matrix 
trans_mat = np.array([[-0.01092, -0.03953,  0.99916,  0.05135],   
                      [-0.99993, -0.00460, -0.01111, -0.02978],  
                      [0.00503,  -0.99921, -0.03948, -0.01958], 
                      [0.00000,   0.00000,  0.00000,  1.00000]])


#Calculating field of view
fov_x = 2*np.arctan2(w,(2*fx))
fov_y = 2*np.arctan2(h,(2*fy))

#Initial masking
min_x = 0.2 #min distance to keep points
max_x = 60 #max distance to keep points
x = points[:,0] #3dimensional decomposition of lidar points
y = points[:,1]
z = points[:,2]

#determining angle of each beam in pointcloud both horizontal and vertical angle
theta_x = np.arctan2(y,x) #2d angles of points correlating to angles of image pixels
theta_y = np.arctan2(z,x)

# filter out points outside the fov angles of the 
### Code for filtering points outside the fov of the camera
positive_mask = np.where((np.abs(theta_x) < (fov_x/2)) & (np.abs(theta_y)<(fov_y/2)) & (x<max_x) & (x > min_x)) #filtering mask
points = np.squeeze(points[positive_mask,:]) #filtered points

r_new,c_new = np.shape(points) #shape of filtered array

### Code for projecting the point cloud onto the image plane ###
extend_homogenous = np.ones((r_new,1)) #creating homogenous extender (r_new,1)

points_homogenous = np.hstack((points,extend_homogenous)) #homogenous array (r_new,4)

points_transformed = points_homogenous@np.linalg.inv(trans_mat).T #transforming array to camera frame
points_transformed_out = points_transformed[:,:3] # discarding 4th dimension for visualization

points_proj = points_transformed@proj_mat.T #initial projection of points to image plane
x = np.divide(points_proj[:,0],points_proj[:,2]) #normalizing to fit with image size
y = np.divide(points_proj[:,1],points_proj[:,2])
z = extend_homogenous

points_out = np.column_stack((x,y,z)) # extending array to 3d to visualize as pointcloud (not necessary)
points_out_idx = np.floor(points_out).astype(int) #rounding to correspoinding pixel in image
positive_mask = np.where((points_out_idx[:,0] < w) & (points_out_idx[:,1] < h) & (points_out_idx[:,0] > 0) &(points_out_idx[:,1] > 0) ) #filtering mask

points_out = np.squeeze(points_out[positive_mask,:])
points_out_idx = np.squeeze(points_out_idx[positive_mask,:])
points_transformed_out = np.squeeze(points_transformed_out[positive_mask,:])

idx_x = points_out_idx[:,0]
idx_y = points_out_idx[:,1]
print(idx_x)
colors = np.array(img[idx_y,idx_x])/255 #using pixel index to determine colors of points
colors = np.vstack((colors,colors,colors)).T #copying value due to image being monocolor

#visualization
pcd_out = o3d.geometry.PointCloud()
pcd_out.points = o3d.utility.Vector3dVector(points_transformed_out)
pcd_out.colors = o3d.utility.Vector3dVector(colors)
print('calculating diameter')
diameter = np.linalg.norm(
    np.asarray(pcd_out.get_max_bound()) - np.asarray(pcd_out.get_min_bound()))
camera = [0, 0, 0]
radius = diameter*8000
print('generating mesh')
_, pt_map = pcd_out.hidden_point_removal(camera, radius)
print('filtering out hidden points')
pcd_out = pcd_out.select_by_index(pt_map)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_out,mesh_frame],zoom=0.1,
                                  front=[0., 0., -1.],
                                  lookat=[0, 0., 1],
                                  up=[0., -1., 0.])
#points = np.asarray(pcd_out.points)
#print(points.shape)

#o3d.io.write_point_cloud("/home/jonathan/isaac_calib_preprocessed/isaac_calib_1_calibrated.ply",
#                          pcd_out, write_ascii=True)