import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import numpy as np
from cv_bridge import CvBridge
global k
global i
k = 0
i = 0
global bridge
bridge = CvBridge()
class Subscriber(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = cv_bridge.CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        global k
        global i
        global bridge
        try:
            k = k+1
            if k % 40 == 0:
                i = i+1
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                print('data = ',cv_image)
                cv2.imwrite('/home/jonathan/Reconstruction_Images/Workspace_2/Images_1/image_' + str(i) + '.png',cv_image)
            
            
        except Exception as e:
            self.get_logger().error('Error processing image: %s' % str(e))


def main(args=None):
    rclpy.init(args=args)
    image_saver = Subscriber()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()