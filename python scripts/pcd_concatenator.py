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
    temp.transform(transform[i])
    if i == 0:
        pcd_out.points = temp.points
        pcd_out.colors = temp.colors
    else:
         pcd_out.points.extend(temp.points)
         pcd_out.colors.extend(temp.colors)
transform_out = np.array([[-1.0, -0.0, 0.0, 0.0],   
                          [0.0, -1.0, 0.0, 0.0],   
                          [0.0, 0.0, 1.0, 0.0],   
                          [0.0, 0.0, 0.0, 1.0]])
pcd_out.transform(transform_out)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_out,mesh_frame],zoom=0.15,
                                  front=[-1., 0., 0.],
                                  lookat=[20, 0., 2],
                                  up=[0., 0., 1.])
    
