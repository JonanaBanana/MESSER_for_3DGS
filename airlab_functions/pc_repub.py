import rclpy
import numpy as np
import ros2_numpy as rnp
from array import array

from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
class Republisher(Node):

    def __init__(self):
        super().__init__('pc_rebpublisher')
        self.publisher = self.create_publisher(PointCloud2,'point_cloud_repub',10)
        #pointcloud subscriber
        self.sub_pc = self.create_subscription(
            PointCloud2,
            'point_cloud',
            self.listener_callback,
            10)
        self.sub_pc  # prevent unused variable warning


    def listener_callback(self, msg):    
        h = msg.height
        w = msg.width
        p = msg.point_step
        intensity = np.ones((w,4))*200
        data = np.array(msg.data)
        data = np.reshape(data,(w,p))
        data = np.hstack((data,intensity))
        dl = np.shape(data)
        dl = dl[0]*dl[1]
        data = np.reshape(data,dl)
        data = data.astype(np.uint8)
        data = array('B', data)
        
        
        fields = [PointField(name='x', offset=0, datatype=7, count=1),
        PointField(name='y', offset=4, datatype=7, count=1),
        PointField(name='z', offset=8, datatype=7, count=1),
        PointField(name='intensity', offset=12, datatype=7,count=1)]
        msg.fields = fields 
        msg.point_step = 16 #len(fields) * 4
        msg.data = data
        self.publisher.publish(msg)
        
            
            
            
            

def main(args=None):
    rclpy.init(args=args)
    node = Republisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
