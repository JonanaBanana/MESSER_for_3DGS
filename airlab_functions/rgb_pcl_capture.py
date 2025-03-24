import rclpy
import os
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2

from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv_bridge
from cv_bridge import CvBridge
import cv2

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
global td
td = 15
global min_points
min_points = 7000
global main_path
global transform_path
global image_path
global pcl_path
main_path = '/home/jonathan/Reconstruction/test_stage_chessboard/'
transform_path = os.path.join(main_path,'transformations.csv')
image_path = os.path.join(main_path,'input/')
pcl_path = os.path.join(main_path,'pcd/')

bridge = CvBridge()
class TimeSyncNode(Node):
    def __init__(self):
        super().__init__('image_saver')
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        if not os.path.isdir(pcl_path):
            os.makedirs(pcl_path)
        
        self.bridge = cv_bridge.CvBridge()
        self.sub_rgb = Subscriber(self,Image,'/rgb')
        self.sub_pcl = Subscriber(self,PointCloud2,'/point_cloud')
        self.target_frame = self.declare_parameter(
          'target_frame', 'camera_frame').get_parameter_value().string_value
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        queue_size = 2
        max_delay = 0.01
        self.time_sync = ApproximateTimeSynchronizer([self.sub_rgb,self.sub_pcl],
                                                     queue_size, max_delay)
        self.time_sync.registerCallback(self.SyncCallback)


    def SyncCallback(self, rgb, pcl):
        global i
        global k
        global transf_out
        global main_path
        global transform_path
        global image_path
        global pcl_path
        global td
        #converting pointcloud to open3d format
        if i>td:
            pcl_data = rnp.numpify(pcl)
            flag1, pcd_out = self.Point_Cloud_Handler(pcl_data['xyz'])
            if flag1 == True:
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
                    flag2 = True
                    
                except TransformException as ex:
                    flag2 = False
                    self.get_logger().info(
                        f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                    return
                
                if flag2==True:
                    flag3 = o3d.io.write_point_cloud(pcl_path + 'pcd_' + str(k) + '.pcd', pcd_out, write_ascii=True)
                    if flag3==True:
                        try:
                            transf_out = np.vstack((transf_out,transf))
                            print("Caught synchronized rgd_pcl pair number %d!" %k)
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
                        cv2.imwrite(image_path + 'img_' + str(k) + '.jpg',cv_rgb)
                    
                        k = k+1
                        i = 0
                    else:
                        print('Error saving the pointcloud...')
                else:
                    print("Error with transform lookup...")
            else:
                print("Not enough points in point cloud...")
        i = i+1
        
    def Point_Cloud_Handler(self, points):
        global min_points
        #("Processing Point Cloud data")
        ### Camera information and tra
        fx = 1108.5125 #focal lengths fx and fy, here they are the same 
        fy = fx
        #height and width in pixels of camera image
        h = 720
        w = 1280
        
        #Calculating field of view
        fov_x = 2*np.arctan2(w,(2*fx))
        fov_y = 2*np.arctan2(h,(2*fy))

        #Initial masking
        min_x = 3 #min distance to keep points
        max_x = 100 #max distance to keep points
        x = points[:,0] #3dimensional decomposition of lidar points
        y = points[:,1]
        z = points[:,2]

        #determining angle of each beam in pointcloud both horizontal and vertical angle
        theta_x = np.arctan2(y,x) #2d angles of points correlating to angles of image pixels
        theta_y = np.arctan2(z,x)

        # filter out points outside the fov angles of the 
        ### Code for filtering points outside the fov of the camera
        positive_mask = np.where((np.abs(theta_x) < (fov_x/2)*0.95) & (np.abs(theta_y)<(fov_y/2)*0.95) & (x<max_x) & (x > min_x)) #filtering mask
        _,M = np.shape(positive_mask)
        #print('positive_matches =',np.sum(positive_mask))
        pcd_out = o3d.geometry.PointCloud()
        if M > min_points:
            print('Found',M,'points!')
            #print('Good scan!')
            flag1 = True
            points = np.squeeze(points[positive_mask,:]) #filtered points

            #visualization
            pcd_out.points = o3d.utility.Vector3dVector(points)
        else:
            #print("Not enough points in the scan")
            flag1 = False
        return flag1, pcd_out


def main(args=None):
    rclpy.init(args=args)
    
    image_saver = TimeSyncNode()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()