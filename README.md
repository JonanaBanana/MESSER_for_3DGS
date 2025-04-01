# 3DGS in feature poor environments

**NOTE: ** _This is a package for testing and developing 3D reconstruction algorithms from RGB, LiDAR and odometry/pose information, specifically for use in feature poor environments where structure from motion using RGB images is not viable. This package is unstable and poorly optimized since it is in very early stages of development as of february 10th, 2025._

## Requirements:

- Open3d
- Colmap (only for converting to gaussian splatting format)
- Numpy
- ROS2 (Only for use for point cloud map generation and odometry information.)
- FAST-LIO (Same as for ROS2)

## With custom data collection:

### First step:

Make sure each requirement is met. I am not sure if specific versions can cause issues, but i do not think so.

### Second step:

Clone into your workspace. If you want to use ROS2 functionalities, clone into ROS2 workspace.

```
cd your_workspace
git clone https://github.com/JonanaBanana/airlab_functions.git
```

If it is a ros2 workspace, make sure to use colcon build.

### Third step - Data collection:

To collect data make sure both FAST-LIO and this package is working.
first run:

```
ros2 run airlab_functions fast_lio_capture
```

then run:

```
ros2 launch fast_lio mapping.launch.py config_file:=your_config_file.yaml
```

This will make sure both FAST_LIO and the data gathering script is aligned, since the fast_lio_capture only runs once fast_lio starts running.

**NOTE:** A few things might need to be adjusted. In airlab*functions open the file fast_lio_img_tranf_capture.py*. In here you can change the variables where it says **Change These**.

- image_topic: the topic publishing the image_topic. Change to your image topic
- odometry_topic: the topic publishing the odometry topic. FAST_LIO publishes on /Odometry as standard.
- main_path: The main folder for saving data. Does not need to exist, and will just be created if it doesn't. However, make sure the path is changed to make sense on your device.
- td: Provides a cooldown once a synchronized pair is found. Change this or set it to 0 to change or disable the cooldown.
- queue_size: The queue size of the approximate synchronizer.
- max_delay: The max delay between the image_topic and odometry_topic. If no data is saved, consider increasing the max delay. This may however cause bad matching between image and odometry, leading to worse reconstruction results.

In your fast_lio folder make sure to create a config file with the specs of your LiDAR and the relative transform between the LiDAR and IMU. Also make sure that fast_lio saves the scanned map once the it is closed. It should save the output in FAST_LIO/PCD/scans.pcd

**Once data has been collected, copy the scans.pcd file into the main path of the folder where the images are stored. This is important**
