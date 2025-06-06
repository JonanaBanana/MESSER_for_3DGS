import rclpy
import os
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge
import cv2

from datetime import datetime 

import numpy as np

from scipy.spatial.transform import Rotation

global i
global k
global td
global image_topic
global odometry_topic
global main_path
global transform_path
global image_path
global rotate_image
global img
global img_time
global max_time_diff
global rest_delay
global last_odom_time

############################ Change These #########################
main_path = get_package_share_directory('messer_for_3dgs')
main_path = os.path.join(main_path,'../../captured_data/')
print('Saving captured data at path:',main_path)
max_time_diff = 25 #milliseconds
rest_delay = 500 #milliseconds
rotate_image = False
image_topic = '/rgb'
odometry_topic = '/Odometry'

####################### DO NOT CHANGE THESE ######################
i = 0
k = 0
last_odom_time = datetime.now()
transform_path = os.path.join(main_path,'transformations.csv')
image_path = os.path.join(main_path,'input/')
##################################################################


bridge = CvBridge()


class ImageNode(Node):
    global image_topic
    global odometry_topic

    def __init__(self):
        super().__init__('image_saver')
        
        self.sub_rgb = self.create_subscription(Image,image_topic,self.image_callback,1)
        self.sub_rgb
        self.sub_odom = self.create_subscription(Odometry,odometry_topic,self.odometry_callback,1)
        self.sub_odom


    def image_callback(self,image_msg):
        global rotate_image
        global img
        global img_time
        img_time = datetime.now()
        img = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        if rotate_image == True:
            img = cv2.rotate(img, cv2.ROTATE_180)
    
    def odometry_callback(self,odom):
        global img
        global img_time
        global transform_path
        global transf_out
        global image_path
        global max_time_diff
        global k
        global rest_delay
        global last_odom_time
        odom_time = datetime.now()
        time_difference = (odom_time - img_time).total_seconds() * 10**3 #milliseconds
        if time_difference < max_time_diff:
            rest_timer = (odom_time - last_odom_time).total_seconds() * 10**3 #milliseconds
            last_odom_time = odom_time
            if rest_timer > rest_delay:
                position_x = odom.pose.pose.position.x
                position_y = odom.pose.pose.position.y
                position_z = odom.pose.pose.position.z
                quat_x = odom.pose.pose.orientation.x
                quat_y = odom.pose.pose.orientation.y
                quat_z = odom.pose.pose.orientation.z
                quat_w = odom.pose.pose.orientation.w
                #creating transformation matrix from frame transform information
                tf_t = np.array([position_x, position_y,position_z])
                tf_r = np.array([quat_x, quat_y, quat_z, quat_w])
                r = Rotation.from_quat(tf_r)
                transf = np.eye(4)
                transf[:3,:3]=r.as_matrix()
                transf[:3,3]=tf_t
                try:
                    transf_out = np.vstack((transf_out,transf))
                    print("Caught synchronized rgb-odometry pair number %d!" %k, ', time difference:',time_difference,'ms.')
                    #print("Transform: " ,transf)
                    np.savetxt(transform_path, transf_out, delimiter=",")
                
                except Exception as e:
                    print(e)
                    print("Creating new transformation matrix list")
                    transf_out = transf
                    #print("Transform: ", transf)
                    np.savetxt(transform_path, transf_out, delimiter=",")
                    
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
                    
                cv2.imwrite(image_path + im_string,img)
                
                k = k + 1
        



def main(args=None):
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    rclpy.init(args=args)
    
    image_saver = ImageNode()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()