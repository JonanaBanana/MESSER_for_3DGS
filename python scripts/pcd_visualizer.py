import open3d as o3d
import numpy as np
pcd = o3d.io.read_point_cloud("/home/jonathan/PCDs/wings_gt_raw.pcd")
pcd_ds = pcd.uniform_down_sample(3)
#pcd_ds = pcd_ds.farthest_point_down_sample(100000)
pcd_out = o3d.geometry.PointCloud()
pcd_out.points = pcd_ds.points
pcd_out.estimate_normals()
pcd.out.remove_statistical_outlier()
#radii = [0.005, 0.01, 0.02, 0.04]
#rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#               pcd, o3d.utility.DoubleVector(radii))

o3d.visualization.draw_geometries([pcd_out],zoom=0.7,
                                  front=[-1, -0.6, 0.1],
                                  lookat=[0, 0., 140.],
                                  up=[0., 0., 1.])
#points = np.asarray(pcd_out.points)
#print(points.shape)

#o3d.io.write_point_cloud("/home/jonathan/PCDs/housing_gt_ds.pcd",
#                         pcd_out, write_ascii=True)