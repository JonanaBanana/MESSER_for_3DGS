import rclpy
import os
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from message_filters import Subscriber, ApproximateTimeSynchronizer

from cv_bridge import CvBridge
import cv2

import numpy as np

from scipy.spatial.transform import Rotation

global i
global k
global simy
global td
global image_topic
global odometry_topic
global main_path
global transform_path
global image_path
global queue_size
global max_delay
global invert_image

############################ Change These #########################
image_topic = '/rgb'
odometry_topic = '/Odometry'
main_path = '/home/jonathan/Reconstruction/test_stage_warehouse_custom'  

invert_image = False
sim = True

if sim == False:
    td = 2
    queue_size = 100
    max_delay = 10
    image_topic = '/camera/color/image_raw'
    odometry_topic = '/Odometry'

    
else:
    td = 8
    queue_size = 10
    max_delay = 0.1
    image_topic = '/rgb'
    odometry_topic = '/Odometry'

##################################################################

####################### DO NOT CHANGE THESE ######################
i = 0
k = 0
transform_path = os.path.join(main_path,'transformations.csv')
image_path = os.path.join(main_path,'input/')
##################################################################


bridge = CvBridge()


class ImageNode(Node):
    global image_topic
    global odometry_topic
    global queue_size
    global max_delay

    def __init__(self):
        super().__init__('image_saver')
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        
        self.sub_rgb = Subscriber(self,Image,image_topic)
        self.sub_odom = Subscriber(self,Odometry,odometry_topic)
        
        self.time_sync = ApproximateTimeSynchronizer([self.sub_rgb,self.sub_odom],
                                                     queue_size, max_delay)
        self.time_sync.registerCallback(self.SyncCallback)


    def SyncCallback(self, rgb, odom):
        global i
        global k
        global transform_path
        global transf_out
        global image_path
        global td
        global invert_image
        #converting pointcloud to open3d format
        if i>td:
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
                print("Caught synchronized rgb-odometry pair number %d!" %k)
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
            if invert_image == True:
                cv_rgb = cv2.rotate(cv_rgb, cv2.ROTATE_180)
                
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
            i = i+1



def main(args=None):
    rclpy.init(args=args)
    
    image_saver = ImageNode()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()