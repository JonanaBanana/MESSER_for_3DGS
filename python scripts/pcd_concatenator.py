import open3d as o3d
import numpy as np
import csv
with open('/home/jonathan/Reconstruction/rgb_pcl_folder/transformations.csv', 'r') as file:
    reader = csv.reader(file)
    transform_data = []
    for row in reader:
        row = [float(i) for i in row]
        transform_data.append(row)
transform_data = np.array(transform_data)
transform = np.reshape(transform_data,(-1,4,4))
N,_,_ = np.shape(transform)
pcd_out = o3d.geometry.PointCloud()
for i in range(N):
    temp = o3d.io.read_point_cloud("/home/jonathan/Reconstruction/rgb_pcl_folder/pcl/pcl_"+str(i)+".pcd")
    points = np.asarray(temp.points)
    r,c = np.shape(points) #shape of filtered array
    ### Code for projecting the point cloud onto the image plane ###
    extend_homogenous = np.ones((r,1)) #creating homogenous extender (r_new,1)
    points_homogenous = np.hstack((points,extend_homogenous))
    points_transformed = points_homogenous@transform[i].T
    if i == 0:
        pcd_out.points = o3d.utility.Vector3dVector(points_transformed[:,:3])
        pcd_out.colors = temp.colors
    else:
         pcd_out.points.extend(o3d.utility.Vector3dVector(points_transformed[:,:3]))
         pcd_out.colors.extend(temp.colors)
transform_out = np.array([[-1.0, -0.0, 0.0, 0.0],   
                          [0.0, -1.0, 0.0, 0.0],   
                          [0.0, 0.0, 1.0, 0.0],   
                          [0.0, 0.0, 0.0, 1.0]])
pcd_out.transform(transform_out)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_out,mesh_frame],zoom=0.1,
                                  front=[0., 0., -1.],
                                  lookat=[0, 0., 1],
                                  up=[0., -1., 0.])
    
