import open3d as o3d
import numpy as np
import csv
import os
import cv2
from copy import deepcopy
main_path = '/home/jonathan/Reconstruction/test_stage_chessboard/'
transform_path = os.path.join(main_path,'transformations.csv')
pcd_path = os.path.join(main_path,'pcd/')
img_path = os.path.join(main_path,'input/')
filtered_path = os.path.join(main_path,'filtered_mask/')
out_path = os.path.join(main_path,'reconstructed.pcd')


mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
mesh_frame_proj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
pcd_temp = o3d.geometry.PointCloud()
pcd_proj = o3d.geometry.PointCloud()


if not os.path.isdir(filtered_path):
            os.makedirs(filtered_path)            

# If in Isaac Sim, set to TRUE
sim = True
# Isaac sims renderer is unstable at edges of image, especially during motion.
# with sim = True, the 10% edges of the image is discarded.

# Enable to filter points in locations with high depth variance
nearest_neighbor_filtering = False

# Enable to display dark pixels in depth filter image (most useful in test environment with dark background)
display_dark_pixels = False

with open(transform_path, 'r') as file:
    reader = csv.reader(file)
    transform_data = []
    for row in reader:
        row = [float(i) for i in row]
        transform_data.append(row)
transform_data = np.array(transform_data)
transform = np.reshape(transform_data,(-1,4,4))
N,_,_ = np.shape(transform)
pcd_out = o3d.geometry.PointCloud()

### Camera information and transformation matrix
fx = 1108.5125 #focal lengths fx and fy, here they are the same 
fy = fx
px = 640 #principal point offset (center of the image plane relative to sensor corner) in x
py = 360 # ppo for y    
s = 0 #skew
#height and width in pixels of camera image
h = 720
w = 1280

#The camera 3x4 camera projection matrix
proj_mat = np.array([[fx, s, px, 0],
                    [0, fy, py, 0],
                    [0, 0, 1, 0]])

#4x4 transformation matrix to transform 3d point from camera frame to LiDAR frame
#This is refined from the calibration result, knowing that the transforms in the simulation environment are much simpler
#To do the inverse transformation simply invert the matrix 
trans_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])


for i in range(N):
    if i%10==0:
        print("Processing pointcloud",i,'/',N)
    img = np.asarray(o3d.io.read_image(img_path+'/img_'+str(i)+'.jpg'))
    pcd = o3d.io.read_point_cloud(pcd_path+'/pcd_'+str(i)+'.pcd')
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    camera = [0, 0, 0]
    radius = diameter*250
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd = pcd.select_by_index(pt_map)
    pcd_2d = deepcopy(pcd)
    points = pcd.points
    r,_ = np.shape(points) #shape of filtered array

    ### Code for projecting the point cloud onto the image plane ###
    extend_homogenous = np.ones((r,1)) #creating homogenous extender (r_new,1)

    points_homogenous = np.hstack((points,extend_homogenous)) #homogenous array (r_new,4)

    points_transformed = points_homogenous@np.linalg.inv(trans_mat).T #transforming array to camera frame
    points_transformed_out = points_transformed[:,:3] # discarding 4th dimension for visualization

    points_proj = points_transformed@proj_mat.T #initial projection of points to image plane
    x = np.divide(points_proj[:,0],points_proj[:,2]) #normalizing to fit with image size
    y = np.divide(points_proj[:,1],points_proj[:,2])
    z = extend_homogenous

    points_out = np.column_stack((x,y,z)) # extending array to 3d to visualize as pointcloud (not necessary)
    points_out_idx = np.round(points_out).astype(int) #rounding to corresponding pixel in image

    if sim == True:
        positive_mask = np.where((points_out_idx[:,0] < w*0.98) & (points_out_idx[:,1] < h*0.98) & (points_out_idx[:,0] > w*0.02) &(points_out_idx[:,1] > h*0.02) ) #filtering mask
    else:
        positive_mask = np.where((points_out_idx[:,0] < w) & (points_out_idx[:,1] < h) & (points_out_idx[:,0] > 0) &(points_out_idx[:,1] > 0) ) #filtering mask


    points_out = np.squeeze(points_out[positive_mask,:])
    points_out_idx = np.round(points_out).astype(int) #rounding to corresponding pixel in image
    points_transformed_out = np.squeeze(points_transformed_out[positive_mask])
    
    idx_x = points_out_idx[:,0]
    idx_y = points_out_idx[:,1]
    colors_out = np.array(img[idx_y,idx_x])/255 #using pixel index to determine colors of points
    
    if nearest_neighbor_filtering == True:
        #Initially draw the pointcloud on top of the image
        depth = points_transformed_out[:,2]
        d_max = np.max(depth)
        d_min = np.min(depth)
        depth_clipped = (((points_transformed_out[:,2]-d_min)/(d_max-d_min))*255).astype(np.uint8)
        depth_colored = np.squeeze(cv2.applyColorMap(depth_clipped, cv2.COLORMAP_JET))
        pixel_data = np.column_stack((points_out_idx[:,0],points_out_idx[:,1]))
        data = np.hstack((pixel_data,depth_colored)).astype(int)
        circle_img = deepcopy(img)
        circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2RGB) 
        for (x, y, r,g,b) in data:
            cv2.circle(circle_img, (x, y), 2, (int(b),int(g),int(r)), -1)
        
        #filter points based on mean nearest neighbor distance in 3D using OPEN3D
        pcd.points = o3d.utility.Vector3dVector(points_transformed_out)
        pcd.colors = o3d.utility.Vector3dVector(colors_out)
        pcd_2d.points = o3d.utility.Vector3dVector(points_out_idx)
        pcd_2d.colors = o3d.utility.Vector3dVector(colors_out)
        _, filter_3d = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        _, filter_2d = pcd_2d.remove_radius_outlier(nb_points=20, radius=22)
        filter = np.intersect1d(filter_3d,filter_2d).astype(int)
        pcd_2d = pcd_2d.select_by_index(filter, invert=True)
        points_2d = np.asarray(pcd_2d.points)[:,:2].astype(int)
        
        #print(filter)
        points_transformed_out = points_transformed_out[filter,:]
        points_out = points_out[filter,:]
        points_out_idx = points_out_idx[filter,:]
        colors_out = colors_out[filter,:]
        for (x, y) in points_2d:
            cv2.circle(circle_img, (x, y), 5, (0,255,0), -1)
        cv2.imwrite(filtered_path+'/filtered_img_'+str(i)+'.jpg',circle_img)

    if display_dark_pixels == True:
        intensity = np.mean(colors_out,axis=1)
        dark_pixels = np.where(intensity<0.1)
        _,K = np.shape(dark_pixels)
        #temporary removal of dark pixels
        K = 0
        if K>0:
            print('dark_pixels in pointcloud',i,':',K)
            dark_pixel_idx = np.squeeze(points_out_idx[dark_pixels,:])
            if K>1:
                
                dark_pixel_idx = dark_pixel_idx
                for l in range(K):
                    cv2.circle(circle_img, (dark_pixel_idx[l,0], dark_pixel_idx[l,1]), 5, (0,0,255), -1)
            else:
                cv2.circle(circle_img, (dark_pixel_idx[0], dark_pixel_idx[1]), 5, (0,0,255), -1)
            cv2.imwrite(filtered_path+'/filtered_img_'+str(i)+'.jpg',circle_img)
    
    pcd_temp.points = o3d.utility.Vector3dVector(points_transformed_out)
    pcd_temp.colors = o3d.utility.Vector3dVector(colors_out)
    pcd_temp.transform(transform[i])
    
    #_, ind = pcd_temp.remove_radius_outlier(nb_points=4, radius=1)
    #pcd_temp = pcd_temp.select_by_index(ind)
    
    
    if i == 0:
        pcd_out.points = pcd_temp.points
        pcd_out.colors = pcd_temp.colors
    else:
         pcd_out.points.extend(pcd_temp.points)
         pcd_out.colors.extend(pcd_temp.colors)
print('filtering points...')
_,ind= pcd_out.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.2)
pcd_out = pcd_out.select_by_index(ind)
pcd_out = pcd_out.voxel_down_sample(voxel_size=0.04)    
print('filtering done!')
o3d.io.write_point_cloud(out_path,
                          pcd_out, write_ascii=True)
print("Saved PCD!")
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_out,mesh_frame],zoom=0.2,
                                  front=[0., 0., -1.],
                                  lookat=[0., -2., 20.],
                                  up=[0., -1., 0.])

