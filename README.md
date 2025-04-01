# 3DGS in feature poor environments

**NOTE:** _This is a package for testing and developing 3D reconstruction algorithms from RGB, LiDAR and odometry/pose information, specifically for use in feature poor environments where structure from motion using RGB images is not viable. This package is unstable and poorly optimized since it is in very early stages of development as of february 10th, 2025._

## Requirements:

- Open3d
- Colmap (only for converting to gaussian splatting format)
- Numpy
- ROS2 (Only for use for point cloud map generation and odometry information.)
- FAST-LIO (Same as for ROS2)

## With custom data collection:

### Step 1:

Make sure each requirement is met. I am not sure if specific versions can cause issues, but i do not think so.

### Step 2:

Clone into your workspace. If you want to use ROS2 functionalities, clone into ROS2 workspace.

```
cd your_workspace
git clone https://github.com/JonanaBanana/airlab_functions.git
```

If it is a ros2 workspace, make sure to use `colcon build`.

### Step 3:

To collect data make sure both FAST-LIO and this package is working.
first run:

```
ros2 run airlab_functions fast_lio_capture
```

then run:

```
ros2 launch fast_lio mapping.launch.py config_file:=your_config_file.yaml
```

This will make sure both FAST_LIO and the data gathering script is aligned, since the `fast_lio_capture` script only runs once fast_lio starts running.

**NOTE:** A few things might need to be adjusted. In `airlab_functions/airlab_functions/` open the file `fast_lio_img_transf_capture.py`. In here you can change the variables where it says **Change These**.

- image_topic: the topic publishing the image_topic. Change to your image topic
- odometry_topic: the topic publishing the odometry topic. FAST_LIO publishes on /Odometry as standard.
- main_path: The main folder for saving data. Does not need to exist, and will just be created if it doesn't. However, make sure the path is changed to make sense on your device.
- td: Provides a cooldown once a synchronized pair is found. Change this or set it to 0 to change or disable the cooldown.
- queue_size: The queue size of the approximate synchronizer.
- max_delay: The max delay between the image_topic and odometry_topic. If no data is saved, consider increasing the max delay. This may however cause bad matching between image and odometry, leading to worse reconstruction results.

In your fast_lio folder make sure to create a config file with the specs of your LiDAR and the relative transform between the LiDAR and IMU. Also make sure that fast_lio saves the scanned map once the it is closed. It should save the output in `FAST_LIO/PCD/scans.pcd`.

**Once data has been collected, copy the scans.pcd file into the main path of the folder where the images are stored. This is important**

### Step 4:

The rest of this process does not use ROS2, but is simply running python scripts to process the data and prepare it for gaussian splatting.
In `airlab_functions/python_scripts` run `generate_color_list.py`

**Note:** A few things might need to be adjusted. These are marked by CONSTANTS and PATHS

- voxel_size: the size of voxels used for voxel downsampling. Voxel-downsampling is important as too many points will cause an overload of VRAM usage in gaussian splatting, so if that is an issue, increase this parameter.
- min_x: the minimum distance of points to be considered for points seen in an image.
- max_x: same as above but maximum distance.
- f, px, py, h, w: all of these are part of the camera intrinsics, so either find the ones relevant to your camera model or determine them using other methods.
- viz: Change to True to see visualization along the way. **This will block the code while visualizing**
- main_path: Change to the main directory used by fast_lio_img_transf_capture.py

_No other variables should need to be changed_
**This might take a long time to run! Progress should be printed**

### Step 5:

In `airlab_functions/python_scripts` run `color_point_cloud.py`

**Note:** A few things might need to be adjusted. These are marked by PATHS

- main_path: Change this to the main directory used by the other scripts.

_No other variables should need to be changed_
**This might take a long time to run! Progress should be printed**

### Step 6:

In `airlab_functions/python_scripts` run `convert_to_colmap.py`

**Note:** A few things might need to be adjusted. These are marked by CONSTANTS and PATHS

- voxel_size: should be the same value as before.
- min_x: the minimum distance of points to be considered for points seen in an image.
- max_x: same as above but maximum distance.
- f, px, py, h, w: all of these are part of the camera intrinsics, so either find the ones relevant to your camera model or determine them using other methods.
- main_path: Change to the main directory used by the other scripts

_No other variables should need to be changed_
**This might take a long time to run! Progress should be printed**

**Note:** This assumes that all images are taken using the same camera so only one camera model is needed. In case this is not true, the code needs to be changed according to colmaps output format of cameras.txt in their github.io

### Step 7:

The last steps are steps in the gaussian splatting package, so follow their steps to ensure everything works. Step 7 requires that you have a working Colmap installation.

```
conda activate gaussian_splatting
cd gaussian-splatting
python convert.py -s /path/to/main_path/ --skip_matching
```

Here the path to the main path should be exactly the same as the main path used in the other python scripts. It is important to include `--skip_matching` as we have created our own matching.

### Step 8:

Finally, train the gaussian splatting!

```
python train.py -s /path/to/main_path/

```

If you have followed the steps to implement the faster training models in the gaussian_splatting github you can use either

```
python train.py -s /path/to/main_path/ --optimizer_type default

```

for an increase in speed or

```
python train.py -s /path/to/main_path/ --optimizer_type sparse_adam

```

for a massive increase in speed.
