import open3d as o3d
import numpy as np
import os
transf_to_camera_frame = False

file_path = os.path.dirname(__file__)  
main_path = os.path.join(file_path, '../../example_stage_warehouse')
pcd_path = os.path.join(main_path, 'scans.pcd')

trans_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
pcd = o3d.io.read_point_cloud(pcd_path)
pcd.remove_non_finite_points()
#pcd,_ = pcd.remove_statistical_outlier(nb_neighbors=10,
#                                        std_ratio=1.5)
#pcd = pcd.voxel_down_sample(voxel_size=0.1)

pcd_out = o3d.geometry.PointCloud()

pcd_out.points = pcd.points
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([mesh_frame,pcd_out],zoom=0.1,
                                  front=[1., 0., 0.],
                                  lookat=[0, 0., 0.],
                                  up=[0.2, 0., 1.])