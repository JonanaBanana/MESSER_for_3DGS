import open3d as o3d
import numpy as np
import csv
import os
main_path = '/home/jonathan/Reconstruction/windmill_full_stage/'
transform_path = os.path.join(main_path,'transformations.csv')
pcd_path = os.path.join(main_path,'pcd/')
img_path = os.path.join(main_path,'input/')
out_path = os.path.join(main_path,'reconstructed.pcd')
sim = True
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
    points_out_idx = np.floor(points_out).astype(int) #rounding to corresponding pixel in image
    idx_x = points_out_idx[:,0]
    idx_y = points_out_idx[:,1]
    ### Coloration can cause wrong colors at edges of objects. How to fix? ###
    colors = np.array(img[idx_y,idx_x])/255 #using pixel index to determine colors of points
    if sim == True:
        positive_mask = np.where((points_out_idx[:,0] < w*0.9) & (points_out_idx[:,1] < h*0.9) & (points_out_idx[:,0] > w*0.1) &(points_out_idx[:,1] > h*0.1) ) #filtering mask
    else:
        positive_mask = np.where((points_out_idx[:,0] < w) & (points_out_idx[:,1] < h) & (points_out_idx[:,0] > 0) &(points_out_idx[:,1] > 0) ) #filtering mask

    points_out = np.squeeze(points_out[positive_mask,:])
    points_out_idx = np.squeeze(points_out_idx[positive_mask,:])
    points_transformed_out = np.squeeze(points_transformed_out[positive_mask])
    colors_out = np.squeeze(colors[positive_mask])
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(points_transformed_out)
    pcd_temp.colors = o3d.utility.Vector3dVector(colors_out)
    pcd_temp.transform(transform[i])
    _, ind = pcd_temp.remove_radius_outlier(nb_points=4, radius=1)
    pcd_temp = pcd_temp.select_by_index(ind)
    
    
    if i == 0:
        pcd_out.points = pcd_temp.points
        pcd_out.colors = pcd_temp.colors
    else:
         pcd_out.points.extend(pcd_temp.points)
         pcd_out.colors.extend(pcd_temp.colors)
         
o3d.io.write_point_cloud(out_path,
                          pcd_out, write_ascii=True)
print("Saved PCD!")
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_out,mesh_frame],zoom=0.2,
                                  front=[0., 0., -1.],
                                  lookat=[0., -2., 20.],
                                  up=[0., -1., 0.])

