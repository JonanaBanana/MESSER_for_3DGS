import open3d as o3d
import numpy as np
pcd = o3d.io.read_point_cloud("/home/jonathan/Reconstruction/windmill_full_stage/reconstructed.pcd")
#pcd_ds = pcd.uniform_down_sample(3)
#pcd_ds = pcd_ds.farthest_point_down_sample(100000)
#pcd_out = o3d.geometry.PointCloud()
#pcd_out.points = pcd_ds.points
#pcd_out.estimate_normals()
#pcd_out.remove_statistical_outlier()
#radii = [0.005, 0.01, 0.02, 0.04]
#rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#               pcd, o3d.utility.DoubleVector(radii))

#o3d.visualization.draw_geometries([pcd_out],zoom=0.7,
#                                  front=[-1, -0.6, 0.1],
#                                  lookat=[0, 0., 140.],
#                                  up=[0., 0., 1.])
#points = np.asarray(pcd_out.points)
#print(points.shape)

#o3d.io.write_point_cloud("/home/jonathan/PCDs/housing_gt_ds.pcd",
#                         pcd_out, write_ascii=True)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd,mesh_frame],zoom=0.2,
                                  front=[0., 0., -1.],
                                  lookat=[0., -2., 20.],
                                  up=[0., -1., 0.])