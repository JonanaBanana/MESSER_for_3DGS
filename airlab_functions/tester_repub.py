import numpy as np

from omni.isaac.core.utils.prims import is_prim_path_valid
import omni.graph.core as og
import rclpy
from rclpy.node import Node
import asyncio
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from builtin_interfaces.msg._time import Time

PCL2_TOPIC_HZ = 20
RENDER_PATH = "/Render/PostProcess/SDGPipeline/"
   
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2

class PCDPublisher(Node):

    def __init__(self):
        super().__init__('pcd_publisher_node')
        self.publisher = self.create_publisher(PointCloud2, 'point_cloud_intensity', 10)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if is_prim_path_valid(RENDER_PATH + "RenderProduct_Isaac_RtxSensorCpuIsaacComputeRTXLidarPointCloud"):
            lidar_compute_node_path = RENDER_PATH + "RenderProduct_Isaac_RtxSensorCpuIsaacComputeRTXLidarPointCloud"   
        for i in range(1,10):
            if is_prim_path_valid(RENDER_PATH + f"RenderProduct_Isaac_0{i}_RtxSensorCpuIsaacComputeRTXLidarPointCloud"):
                lidar_compute_node_path = RENDER_PATH + f"RenderProduct_Isaac_0{i}_RtxSensorCpuIsaacComputeRTXLidarPointCloud"
        print("determined path :",lidar_compute_node_path)
        
        # receiving lidar data from Isaac
        r_arr = og.Controller().node(lidar_compute_node_path).get_attribute("outputs:range").get()
        phi_arr = og.Controller().node(lidar_compute_node_path).get_attribute("outputs:elevation").get()
        theta_arr = og.Controller().node(lidar_compute_node_path).get_attribute("outputs:azimuth").get()
        
        x_arr = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
        y_arr = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
        z_arr = r_arr * np.cos(theta_arr)
        cartesian_arr = np.column_stack((x_arr.value, y_arr.value * (-1), z_arr.value))
        print('Cartesian Array : ',cartesian_arr)
        
        intensity_from_isaac = og.Controller().node(lidar_compute_node_path).get_attribute("outputs:intensity").get()

        # adding intensity in [x,y,z]                                                                                                            
        self.points = np.column_stack((cartesian_arr, intensity_from_isaac))

        ros_time = self.get_clock().now().to_msg()
        self.time = ros_time
        self.pcd = self._create_point_cloud(self.points, self.time, 'Rotating')
        self.publisher.publish(self.pcd)

    def _create_point_cloud(self, points, time, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx4 array of xyz positions and intensity data.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes()
        fields = [
                sensor_msgs.PointField(name='x', offset=0, datatype=ros_dtype, count=1),
                sensor_msgs.PointField(name='y', offset=4, datatype=ros_dtype, count=1),
                sensor_msgs.PointField(name='z', offset=8, datatype=ros_dtype, count=1),
                sensor_msgs.PointField(name='intensity', offset=12, datatype=ros_dtype, count=1),
            ]
        header = std_msgs.Header(stamp=time, frame_id=parent_frame)

        return sensor_msgs.PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 4),  # Every point consists of four float32s.
            row_step=(itemsize * 4 * points.shape[0]),
            data=data
        )

async def my_task_lidar(): 
    publisher = PCDPublisher()
    
    while rclpy.ok():   
        rclpy.spin_once(publisher)
        
        await asyncio.sleep(0.05)
    publisher.unregister()
    publisher = None

frame = 0 
while simulation_app.is_running():

    if frame==48:
        asyncio.ensure_future(my_task_lidar())
     
    simulation_context.step(render=True)
    frame = frame + 1

simulation_context.stop()
simulation_app.close()
    