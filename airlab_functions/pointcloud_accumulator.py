import rclpy
import os
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2

import numpy as np
import ros2_numpy as rnp
import open3d as o3d

from scipy.spatial.transform import Rotation

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener



global i
i = 0
global k
k = 0
global pcl_path
global pcd_temp
global pcd_out
pcd_temp = o3d.geometry.PointCloud()
pcd_out = o3d.geometry.PointCloud()
main_path = '/home/jonathan/Reconstruction/test_stage_chessboard_2/'
pcl_path = os.path.join(main_path,'pcd/')

global trans_mat
trans_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])

class PointCloudNode(Node):
    def __init__(self):
        super().__init__('point_cloud_accumulator')
        if not os.path.isdir(pcl_path):
            os.makedirs(pcl_path)
        

        self.target_frame = self.declare_parameter(
          'target_frame', 'camera_frame').get_parameter_value().string_value
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        
        self.subscription = self.create_subscription(
            PointCloud2,
            '/point_cloud',
            self.listener_callback,
            10)


    def listener_callback(self, pcl):
        global i
        global k
        global pcl_path
        global trans_mat
        global pcd_temp
        global pcd_out
        #converting pointcloud to open3d format
        pcl_data = rnp.numpify(pcl)
        points = pcl_data['xyz']
        time = rclpy.time.Time(seconds=pcl.header.stamp.sec,nanoseconds=pcl.header.stamp.nanosec)
        from_frame_rel = self.target_frame
        to_frame_rel = 'camera_init'
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                time)
            #creating transformation matrix from frame transform information
            tf_t = np.array([t.transform.translation.x, t.transform.translation.y,t.transform.translation.z])
            tf_r = np.array([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])
            r = Rotation.from_quat(tf_r)
            transf = np.eye(4)
            transf[:3,:3]=r.as_matrix()
            transf[:3,3]=tf_t
            flag = True
            
        except TransformException as ex:
            flag = False
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return
        
        if flag==True:
                r,_ = np.shape(points) #shape of filtered array

                ### Code for projecting the point cloud onto the image plane ###
                extend_homogenous = np.ones((r,1)) #creating homogenous extender (r_new,1)

                points_homogenous = np.hstack((points,extend_homogenous)) #homogenous array (r_new,4)

                points_transformed = points_homogenous@np.linalg.inv(trans_mat).T #transforming array to camera frame
                points_transformed_out = points_transformed[:,:3] # discarding 4th dimension for visualization
                pcd_temp.points = o3d.utility.Vector3dVector(points_transformed_out)
                pcd_temp.transform(transf)
                if i == 0:
                    pcd_out.points = pcd_temp.points
                    i = 1
                else:
                    pcd_out.points.extend(pcd_temp.points)
                    N,_ = np.shape(np.asarray(pcd_out.points))
                    if N > 10**7:
                        flag2 = o3d.io.write_point_cloud(pcl_path + 'pcd_' + str(k) + '.pcd', pcd_out, write_ascii=True)
                        if flag2 == True:
                            print("Point cloud growing large, saving accumulated scan and resetting point cloud")
                            i = 0
                            k = k +1
                        else:
                            print("Error saving point cloud")
        else:
            print("Error with transform lookup...")


def main(args=None):
    global pcd_out
    global k
    global pcl_path
    rclpy.init(args=args)
    image_saver = PointCloudNode()
    try:
        rclpy.spin(image_saver)
    except KeyboardInterrupt:
        #Save the current point cloud
        print("\nSaving final point cloud")
        flag = o3d.io.write_point_cloud(pcl_path + 'pcd_' + str(k) + '.pcd', pcd_out, write_ascii=True)
        if flag == True:
            print("Save successfull")
            k = k +1
        else:
            print("Error saving point cloud...")
        print("\nPostprocessing, might take a while...")
        #Post processing of point clouds
        #First, voxel downsample the point clouds
        #Second, add them together
        #Third, voxel downsample the full point cloud
        #Fourth, save the full point cloud
        for i in range(k):
            print("Processing point cloud: "+str(i)+"/"+str(k))
            if i == 0:
                pcd = o3d.io.read_point_cloud(pcl_path+'/pcd_'+str(i)+'.pcd')
                pcd = pcd.voxel_down_sample(voxel_size=0.04)
            else:
                pcd_temp = o3d.io.read_point_cloud(pcl_path+'/pcd_'+str(i)+'.pcd')
                pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.04)
                pcd.points.extend(pcd_temp.points)
        print("Downsampling final point cloud...")
        pcd = pcd.voxel_down_sample(voxel_size=0.04)
        flag = o3d.io.write_point_cloud(pcl_path + 'accumulated_point_cloud.pcd', pcd, write_ascii=True)
        if flag == True:
            print("Successfully saved point cloud")
        else:
            print("Error saving accumulated point cloud...")
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd,mesh_frame],zoom=0.2,
                                          front=[0., 0., -1.],
                                          lookat=[0., -2., 20.],
                                          up=[0., -1., 0.])    
        print("DONE, SHUTTING DOWN")
        pass
    image_saver.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()