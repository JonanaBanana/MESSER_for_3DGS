import numpy as np
import open3d as o3d
import os
from plyfile import PlyData
from copy import deepcopy
from ament_index_python.packages import get_package_share_directory

######################### CONSTANTS ###############################
voxel_size = 0.5
filter_factor = 1.5

viz = True
#################################################################

########################## PATHS ################################
main_path = get_package_share_directory('messer_for_3dgs')
main_path = os.path.join(main_path,'../../captured_data/')
ply_path = os.path.join(main_path,'iteration_30000/point_cloud.ply')
scans_path = os.path.join(main_path,'scans.pcd')
output_folder = os.path.join(main_path,'iteration_31000')
output_path = os.path.join(output_folder,'point_cloud.ply')
#################################################################

if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

plydata = PlyData.read(ply_path)

x = plydata['vertex']['x']
y = plydata['vertex']['y']
z = plydata['vertex']['z']
points = np.vstack((x,y,z)).T
comp_pcd = o3d.io.read_point_cloud(scans_path)
comp_pcd.remove_non_finite_points()
comp_pcd = comp_pcd.voxel_down_sample(voxel_size=voxel_size)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

if viz == True:    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd,comp_pcd],zoom=0.2,
                                    front=[-1., 0., 0.],
                                    lookat=[10., 0., 0.],
                                    up=[0., 0., 1.])
    
n_points = np.squeeze(np.shape(x))
comp_points = np.asarray(comp_pcd.points)
min_dist = np.empty((n_points,1))
print("Progress:")
for i in range(n_points):
    if i%10000 == 0:
        print(str(i)+'/'+str(n_points))
    temp_point = points[i,:]
    dist = np.linalg.norm(temp_point-comp_points,axis=1)
    min_dist[i] = np.min(dist)
filtered_index = np.empty((0,1))
for i in range(n_points):
    if min_dist[i] < voxel_size*filter_factor:
        filtered_index = np.append(filtered_index,i)
filtered_index = filtered_index.astype(int)
#print('shape of input point cloud: ',n_points)
#print('shape of filtered point cloud: ',np.squeeze(np.shape(filtered_index)))


plydata.elements[0].data = plydata.elements[0].data[filtered_index]
plydata.write(output_path)

#print(plydata)

if viz == True:
    points_filtered = points[filtered_index,:]
    print('output array shape: ', np.shape(points_filtered))
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points_filtered)    
    o3d.visualization.draw_geometries([pcd_out],zoom=0.2,
                                    front=[-1., 0., 0.],
                                    lookat=[10., 0., 0.],
                                    up=[0., 0., 1.])
