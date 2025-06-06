import rclpy
import os
from rclpy.node import Node

from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory
from message_filters import Subscriber

from cv_bridge import CvBridge
import cv2

import numpy as np

from scipy.spatial.transform import Rotation

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener



global i
i = 0
global k
k = 0
global td
td = 100
global main_path
global transform_path
global image_path
global pcl_path
main_path = get_package_share_directory('messer_for_3dgs')
main_path = os.path.join(main_path,'../../captured_data/')
transform_path = os.path.join(main_path,'transformations.csv')
image_path = os.path.join(main_path,'input/')

bridge = CvBridge()
class ImageNode(Node):
    def __init__(self):
        super().__init__('image_saver')
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        
        self.target_frame = self.declare_parameter(
          'target_frame', 'Mover').get_parameter_value().string_value
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.listener_callback,
            10)
        self.subscription


    def listener_callback(self, rgb):
        global i
        global k
        global transf_out
        global transform_path
        global image_path
        global td
        #converting pointcloud to open3d format
        if i>td:
            from_frame_rel = self.target_frame
            to_frame_rel = 'odom'
            
            try:
                t = self.tf_buffer.lookup_transform(
                    to_frame_rel,
                    from_frame_rel,
                    rclpy.time.Time())
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
                try:
                    transf_out = np.vstack((transf_out,transf))
                    print("Caught synchronized rgb+transform pair number %d!" %k)
                    #print("Transform: " ,transf)
                    np.savetxt(transform_path, transf_out, delimiter=",")
                
                except Exception as e:
                    print(e)
                    print("Creating new transformation matrix list")
                    transf_out = transf
                    #print("Transform: ", transf)
                    np.savetxt(transform_path, transf_out, delimiter=",")
                
                #saving point cloud
                #saving rgb image and pointcloud
                cv_rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
                if k >= 10:
                    im_string = '/img_000' + str(k) + '.jpg'
                    if k >= 100:
                        im_string = '/img_00' + str(k) + '.jpg'
                        if k >= 1000:
                            im_string = '/img_0' + str(k) + '.jpg'
                            if k >= 10000:
                                im_string = '/img_' + str(k) + '.jpg'
                else:
                    im_string = '/img_0000' + str(k) + '.jpg'
                    
                cv2.imwrite(image_path + im_string,cv_rgb)
            
                k = k+1
                i = 0
            else:
                print("Error with transform lookup...")
        i = i+1



def main(args=None):
    rclpy.init(args=args)
    
    image_saver = ImageNode()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()