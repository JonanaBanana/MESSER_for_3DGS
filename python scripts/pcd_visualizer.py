import open3d as o3d
import numpy as np
transf_to_camera_frame = False

trans_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
pcd = o3d.io.read_point_cloud("/home/jonathan/Reconstruction/test_stage_windmill_custom/scans.pcd")
pcd = pcd.voxel_down_sample(voxel_size=0.1)
#pcd.transform(np.linalg.inv(trans_mat))
pcd_out = o3d.geometry.PointCloud()

#if transf_to_camera_frame == True:
#    points = np.asarray(pcd.points)
#    r,_ = np.shape(points) #shape of filtered array

    ### Code for projecting the point cloud onto the image plane ###
#    extend_homogenous = np.ones((r,1)) #creating homogenous extender (r_new,1)

#    points_homogenous = np.hstack((points,extend_homogenous)) #homogenous array (r_new,4)

#    points_transformed = points_homogenous@np.linalg.inv(trans_mat).T #transforming array to camera frame
#    points_transformed_out = points_transformed[:,:3] # discarding 4th dimension for visualization
#    pcd.points = o3d.utility.Vector3dVector(points_transformed_out)
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
pcd_out.points = pcd.points
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd_out,mesh_frame],zoom=0.2,
                                  front=[-1., 0., 0.],
                                  lookat=[10., -2., 0.],
                                  up=[0., 0., 1.])