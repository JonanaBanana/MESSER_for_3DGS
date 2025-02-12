import rclpy
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
global d_max
d_max = 60

bridge = CvBridge()
class TimeSyncNode(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = cv_bridge.CvBridge()
        self.sub_rgb = Subscriber(self,Image,'/rgb')
        self.sub_pcl = Subscriber(self,PointCloud2,'/point_cloud')
        self.pub_img = self.create_publisher(Image,'/rgb_pcl_viz',10)
        
        queue_size = 10
        max_delay = 0.05
        self.time_sync = ApproximateTimeSynchronizer([self.sub_rgb,self.sub_pcl],
                                                     queue_size, max_delay)
        self.time_sync.registerCallback(self.SyncCallback)


    def SyncCallback(self, rgb, pcl):
        #convering pointcloud to open3d format
        pcl_data = rnp.numpify(pcl)
        self.Point_Cloud_Handler(rgb,pcl_data['xyz'])
        
        
    def Point_Cloud_Handler(self,img, points):
        global d_max
        ### Camera information and tra
        fx = 1108.5125 #focal lengths fx and fy, here they are the same 
        fy = fx
        px = 640 #principal point offset (center of the image plane relative to sensor corner) in x
        py = 360 # ppo for y
        s = 0 #skew
        #height and width in pixels of camera image
        h = 720
        w = 1280

        #The camera 3x4 camera projection matrix
        proj_mat = np.array([[fx, s, px, 0],
                            [0, fy, py, 0],
                            [0, 0, 1, 0]])

        #Rotation matrix for aligning rotation of Camera and LiDAR
        #currently this is an identity matrix since the up-direction of camera and lidar is aligned
        rot_mat = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])

        #4x4 transformation matrix to transform 3d point from camera frame to LiDAR frame
        #This is refined from the calibration result, knowing that the transforms in the simulation environment are much simpler
        #To do the inverse transformation simply invert the matrix 
        trans_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])


        #Calculating field of view
        fov_x = 2*np.arctan2(w,(2*fx))
        fov_y = 2*np.arctan2(h,(2*fy))

        #Initial masking
        min_x = 0.2 #min distance to keep points
        max_x = 60 #max distance to keep points
        x = points[:,0] #3dimensional decomposition of lidar points
        y = points[:,1]
        z = points[:,2]

        #determining angle of each beam in pointcloud both horizontal and vertical angle
        theta_x = np.arctan2(y,x) #2d angles of points correlating to angles of image pixels
        theta_y = np.arctan2(z,x)

        # filter out points outside the fov angles of the 
        ### Code for filtering points outside the fov of the camera
        positive_mask = np.where((np.abs(theta_x) < (fov_x/2)) & (np.abs(theta_y)<(fov_y/2)) & (x<max_x) & (x > min_x)) #filtering mask
        points = np.squeeze(points[positive_mask,:]) #filtered points

        r_new,_ = np.shape(points) #shape of filtered array

        ### Code for projecting the point cloud onto the image plane ###
        extend_homogenous = np.ones((r_new,1)) #creating homogenous extender (r_new,1)

        points_homogenous = np.hstack((points,extend_homogenous)) #homogenous array (r_new,4)

        points_transformed = points_homogenous@np.linalg.inv(trans_mat).T #transforming array to camera frame
        points_transformed_out = points_transformed[:,:3] # discarding 4th dimension for visualization

        points_proj = points_transformed@proj_mat.T #initial projection of points to image plane
        x = np.divide(points_proj[:,0],points_proj[:,2]) #normalizing to fit with image size
        y = np.divide(points_proj[:,1],points_proj[:,2])
        z = extend_homogenous

        points_out = np.column_stack((x,y,z)) # extending array to 3d to visualize as pointcloud (not necessary)
        points_out_idx = np.floor(points_out).astype(int) #rounding to correspoinding pixel in image
        positive_mask = np.where((points_out_idx[:,0] < w) & (points_out_idx[:,1] < h) & (points_out_idx[:,0] > 0) &(points_out_idx[:,1] > 0) ) #filtering mask

        points_out = np.squeeze(points_out[positive_mask,:])
        points_out_idx = np.squeeze(points_out_idx[positive_mask,:])
        points_transformed_out = np.squeeze(points_transformed_out[positive_mask,:])

        idx_x = points_out_idx[:,0]
        idx_y = points_out_idx[:,1]
        depth_clipped = np.clip(points_transformed_out[:,2],a_min=None,a_max=max_x)
        d_max = 60
        depth_clipped = ((depth_clipped/d_max)*255).astype(np.uint8)
        depth_colored = np.squeeze(cv2.applyColorMap(depth_clipped, cv2.COLORMAP_JET))
        r = np.shape(idx_x)
        data = np.column_stack((idx_x,idx_y))
        min_points = 10000
        if r[0] > min_points:
            data = np.hstack((data,depth_colored)).astype(int)
            cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
            for (x, y, r,g,b) in data:
                cv2.circle(cv_image, (x, y), 2, (int(b),int(g),int(r)), -1)
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = img.header
            self.pub_img.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    
    image_saver = TimeSyncNode()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()