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

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener



global i
i = 0
global k
k = 0

bridge = CvBridge()
class TimeSyncNode(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = cv_bridge.CvBridge()
        self.sub_rgb = Subscriber(self,Image,'/rgb')
        self.sub_pcl = Subscriber(self,PointCloud2,'/point_cloud')
        self.target_frame = self.declare_parameter(
          'target_frame', 'Camera').get_parameter_value().string_value
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        queue_size = 10
        max_delay = 0.01
        self.time_sync = ApproximateTimeSynchronizer([self.sub_rgb,self.sub_pcl],
                                                     queue_size, max_delay)
        self.time_sync.registerCallback(self.SyncCallback)


    def SyncCallback(self, rgb, pcl):
        global i
        global k
        global transf_out
        
        if i%25==10:
            time = rclpy.time.Time(seconds=pcl.header.stamp.sec,nanoseconds=pcl.header.stamp.nanosec)
            from_frame_rel = self.target_frame
            to_frame_rel = 'odom'
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
                
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                return
            
            #convering pointcloud to open3d format
            pcl_data = rnp.numpify(pcl)
            img_data = rnp.numpify(rgb)
            flag = self.Point_Cloud_Handler(img_data,pcl_data['xyz'],k)
            if flag==True:
                try:
                    transf_out = np.vstack((transf_out,transf))
                    print("Caught synchronized rgd_pcl pair number %d!" %k)
                    print("Transform: " ,transf)
                    np.savetxt('/home/jonathan/Reconstruction/rgb_pcl_folder/transformations.csv', transf_out, delimiter=",")
                
                except Exception as e:
                    print(e)
                    print("Creating new transformation matrix list")
                    transf_out = transf
                    print("Transform: ", transf)
                    np.savetxt('/home/jonathan/Reconstruction/rgb_pcl_folder/transformations.csv', transf_out, delimiter=",")
                
                #saving point cloud
                #saving rgb image and pointcloud
                cv_rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
                cv2.imwrite('/home/jonathan/Reconstruction/rgb_pcl_folder/rgb/rgb_' + str(k) + '.png',cv_rgb)
                print("Image Saved!")
            
            
                k = k+1
        i = i+1
        
    def Point_Cloud_Handler(self,img, points,k):
        print("Processing Point Cloud data")
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
        min_x = 5 #min distance to keep points
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
        idx_x = points_out_idx[:,0]
        idx_y = points_out_idx[:,1]
        colors = np.array(img[idx_y,idx_x])/255 #using pixel index to determine colors of points

        #visualization
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(points_transformed_out)
        pcd_out.colors = o3d.utility.Vector3dVector(colors)
        flag1 = o3d.io.write_point_cloud('/home/jonathan/Reconstruction/rgb_pcl_folder/pcl/pcl_' + str(k) + '.pcd', pcd_out, write_ascii=True)
        if flag1 == True:
            print("Point Cloud Saved!")
        elif flag1==False:
            print("Error saving pointcloud...")
        return flag1


def main(args=None):
    rclpy.init(args=args)
    
    image_saver = TimeSyncNode()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()