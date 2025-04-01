import numpy as np
import open3d as o3d
import csv
import os
from copy import deepcopy

########################## PATHS ################################
main_path = '/home/jonathan/Reconstruction/test_stage_chessboard_4'
downsampled_path = os.path.join(main_path,'downsampled_point_cloud.pcd')
csv_path = os.path.join(main_path,'point_cloud_color_information.csv')
out_path = os.path.join(main_path,'reconstructed.pcd')
#################################################################

pcd_fl = o3d.io.read_point_cloud(downsampled_path)
pcd = o3d.geometry.PointCloud()
pcd.points = pcd_fl.points
points = np.asarray(pcd.points)
        
with open(csv_path, 'r') as file:
    reader = csv.reader(file)
    data_list = []
    for row in reader:
        row = [float(i) for i in row]
        data_list.append(row)
data_list = np.array(data_list)
data_list = np.reshape(data_list,(-1,4))
idx_list = data_list[:,0].astype(int)
colors_list = data_list[:,1:].astype(float)
R,_ = np.shape(np.asarray(pcd.points))
print("Found ",R,'points in point cloud')
visible_idx = np.sort(np.unique(idx_list))
print(np.min(visible_idx))
print(np.max(visible_idx))
M = np.shape(np.asarray(visible_idx))[0]
print("Visible points from images:",M)
points_out = points[visible_idx,:]
print('visible_idx:',np.shape(visible_idx))
colors_out = np.zeros((R,3))
print('colors_out:',np.shape(colors_out))
print("Generating color for each point cloud")
i = 0
for idx in visible_idx:
    i = i+1
    if i%1000 == 0:
        print("Progress: "+str(i)+"/"+str(M))
    present_colors_idx = np.where(idx_list == idx)
    K = np.shape(np.asarray(present_colors_idx))[1]
    colors = colors_list[present_colors_idx]
    colors = np.sort(colors, axis=0, kind='mergesort')
    colors = np.median(colors,axis=0)
    colors_out[idx,:] = colors

colors_out = colors_out[visible_idx,:]
pcd.points = o3d.utility.Vector3dVector(points_out)
pcd.colors = o3d.utility.Vector3dVector(colors_out)
o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd,mesh_frame],zoom=0.3,
                                  front=[0., 0., -1.],
                                  lookat=[0., -2., 20],
                                  up=[0., -1., 0.])