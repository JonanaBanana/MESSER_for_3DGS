import rclpy
import numpy as np
import open3d as o3d
import ros2_numpy as rnp
import signal

from scipy.spatial.transform import Rotation
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

global pcd
global flag_1
global pc_pose_subscriber
flag_1 = 0
transf = []   
class Subscriber(Node):

    def __init__(self):
        super().__init__('isaac_sim_pose_subscriber')
        
        self.target_frame = self.declare_parameter(
          'target_frame', 'Mover').get_parameter_value().string_value
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        #pointcloud subscriber
        self.sub_pc = self.create_subscription(
            PointCloud2,
            'point_cloud',
            self.listener_callback_pc,
            10)
        self.sub_pc  # prevent unused variable warning


    def listener_callback_pc(self, msg):
        global pcd
        global flag_1
        from_frame_rel = self.target_frame
        to_frame_rel = 'odom'
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return
        tf_t = np.array([t.transform.translation.x, t.transform.translation.y,t.transform.translation.z])
        tf_r = np.array([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])
        r = Rotation.from_quat(tf_r)
        transf = np.eye(4)
        transf[:3,:3]=r.as_matrix()
        transf[:3,3]=tf_t
        
        if transf != []:
            temp = rnp.numpify(msg)
            points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(temp['xyz']))
            points = points.transform(transf)
            if flag_1 == 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = points.points
                self.get_logger().info('RECEIVING DATA')
                flag_1 = 1
            else:
                pcd.points.extend(points.points)
        else:
            self.get_logger().info('Waiting for pose...')
        


def main(args=None):
    global pc_pose_subscriber
    global pcd
    rclpy.init()
    pc_pose_subscriber = Subscriber()
    try:
        rclpy.spin(pc_pose_subscriber)
    except KeyboardInterrupt:
        print("\nPostprocessing, might take a while...")
        print("Removing Statistical outliers...")
        [pcd,_] = pcd.remove_statistical_outlier(8, 0.8)
        print("Done!")
        print("Removing duplicated points...")
        pcd = pcd.remove_duplicated_points()
        print("Done!")
        print("Uniform Downsampling")
        pcd = pcd.uniform_down_sample(3)
        print("Done!")
        print("Estimating Normals...")
        pcd.estimate_normals(fast_normal_computation=False)
        print("Done!")
        o3d.io.write_point_cloud("/home/jonathan/airlab-uav/src/airlab_functions/test.pcd", pcd)
        print("Saved as PCD")
        print("Opening Visualizer")
        o3d.visualization.draw_geometries([pcd])
        print("DONE, SHUTTING DOWN")
        pass
        
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pc_pose_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
