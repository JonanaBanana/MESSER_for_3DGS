# Method for SfM-free Sparse Environment Reconstruction (MESSER)

This package provides a pipeline for SfM-free 3DGS reconstruction optimized towards more coherent reconstructions in large sparse environments such as windmills. The poses of images and points are estimated solely from IMU and LiDAR using FAST-LIO, so there is no dependency on visual features. Thus any sparse environment should be reconstructable as long as you can achieve accurate pose predictions and point clouds. The main focus is on achieving coherent sparse environments by minimizing floating gaussians and overfitting of free space during 3DGS reconstruction.

![alt text](https://github.com/JonanaBanana/MESSER_for_3DGS/blob/main/images/environment_comparison.png?raw=true)
![alt text](https://github.com/JonanaBanana/MESSER_for_3DGS/blob/main/images/windmill_comparison.png?raw=true)
![alt text](https://github.com/JonanaBanana/MESSER_for_3DGS/blob/main/images/filtering.png?raw=true)

**NOTE:** _This is a package for testing and developing 3D reconstruction algorithms from RGB, LiDAR and odometry/pose information, specifically for use in feature poor environments where structure from motion using RGB images is not viable. This package is not fully and optimally developed, but is a work in progress as of May 29th, 2025._

## Requirements:

- Open3d
- Colmap (only for converting to gaussian splatting format)
- Numpy
- ROS2 (Only for use for point cloud map generation and odometry information.)
- FAST-LIO2 (Same as for ROS2)

## Quick Test

To quickly and easily evaluate the performance of the method, a sample dataset is provided in the package as `MESSER_for_3DGS/example_stage_warehouse`.
The folders of the python scripts has been setup to process that example folder by default so running it will perform the reconstruction steps with vizualization. The scripts are not setup to be run from shell, only the ROS2 scripts. Simply run the following scripts in vscode.

### Step 1:

In `MESSER_for_3DGS/python_scripts` run `generate_color_list.py`

In `MESSER_for_3DGS/python_scripts` run `color_point_cloud.py`

In `MESSER_for_3DGS/python_scripts` run `convert_to_colmap.py`

To add additional vizualisation during runtime, set viz=True in `generate_color_list.py`.

Now the data will be prepared for use with 3DGS. Follow step 7-8 in the section below to do the actual gaussian splatting training and evaluation.

## With custom data collection:

### Step 1:

Make sure each requirement is met. I am not sure if specific versions can cause issues, but i do not think so.

### Step 2:

Clone into your workspace. If you want to use ROS2 functionalities, clone into ROS2 workspace.

```
cd your_workspace
git clone https://github.com/JonanaBanana/MESSER_for_3DGS.git
```

If it is a ros2 workspace, make sure to use `colcon build`.

### Step 3:

To collect data make sure both FAST-LIO and this package is working.
first run:

```
ros2 run messer_for_3dgs isaacsim_subscriber
```

then run:

```
ros2 launch fast_lio mapping.launch.py config_file:=your_config_file.yaml
```

This will make sure both FAST_LIO and the data gathering script is aligned, since the `fast_lio_capture` script only runs once fast_lio starts running.
If your image and odometry topics are not time synchronized, you can instead use

```
ros2 run messer_for_3dgs subscriber
```

Which will not try to find synchronized timestamps but will instead just look for sets published within a given timeframe.

**NOTE:** A few things might need to be adjusted. In `MESSER_for_3DGS/ros2/` open the file `isaacsim_subscriber.py`. In here you can change the variables where it says **Change These**.

- image_topic: the topic publishing the image_topic. Change to your image topic.
- odometry_topic: the topic publishing the odometry topic. FAST_LIO publishes on /Odometry as standard.
- main_path: The main folder for saving data. Does not need to exist, and will just be created if it doesn't. By default data is saved in the install folder of the workspace under messer_for_3dgs.
- td: Provides a cooldown once a synchronized pair is found. Change this or set it to 0 to change or disable the cooldown.
- queue_size: The queue size of the approximate synchronizer.
- max_delay: The max delay between the image_topic and odometry_topic. If no data is saved, consider increasing the max delay. This may however cause bad matching between image and odometry, leading to worse reconstruction results.
- invert_image: if your captured images are upside-down, enable invert image.
- sim: can be set to true to also capture ground truth data in IsaacSim if you have ground truth data available. Currently does not work as intended.

In your fast_lio folder make sure to create a config file with the specs of your LiDAR and the relative transform between the LiDAR and IMU. Also make sure that fast_lio saves the scanned map once the it is closed. It should save the output in `FAST_LIO/PCD/scans.pcd`.

**Once data has been collected, copy the scans.pcd file into the main path of the folder where the images are stored. This is important**

### Step 4:

The rest of this process does not use ROS2, but is simply running python scripts to process the data and prepare it for gaussian splatting.
In `MESSER_for_3DGS/python_scripts` run `generate_color_list.py`

**Note:** A few things might need to be adjusted. These are marked by CONSTANTS and PATHS

- use_gt_pose: if you captures ground truth poses previously, enable this to use ground truth poses instead of fast-lio poses.
- fill_background: enable to add a spherical backround around your scene. This helps reduce floating gaussians for 3DGS reconstruction. Adjust sphere_center, sphere_radius and sphere_num_pts to fit the scene.
- hidden_point_removal_factor: adjust to change how severely to remove hidden points. Larger values provide sharper edges, but might color points that are not visible in regions with low density. Can be tuned with viz=True to see the impact.
- voxel_size: the size of voxels used for voxel downsampling. Voxel-downsampling is important as too many points will cause an overload of VRAM usage in gaussian splatting, so if that is an issue, increase this parameter.
- min_x: the minimum distance of points to be considered for points seen in an image.
- max_x: same as above but maximum distance.
- f, px, py, h, w: all of these are part of the camera intrinsics, so either find the ones relevant to your camera model or determine them using other methods.
- viz: Change to True to see visualization along the way. **This will block the code while visualizing**
- main_path: must be the same main directory used by isaacsim_subscriber.py.

_No other variables should need to be changed_

**This might take a long time to run!**

### Step 5:

In `MESSER_for_3DGS/python_scripts` run `color_point_cloud.py`

**Note:** A few things might need to be adjusted. These are marked by PATHS

- main_path: Change this to the main directory used by the other scripts.

_No other variables should need to be changed_

**This might take a long time to run!**

### Step 6:

In `MESSER_for_3DGS/python_scripts` run `convert_to_colmap.py`

**Note:** A few things might need to be adjusted. These are marked by CONSTANTS and PATHS

- voxel_size: should be the same value as before.
- min_x: the minimum distance of points to be considered for points seen in an image.
- max_x: same as above but maximum distance.
- f, px, py, h, w: all of these are part of the camera intrinsics, so either find the ones relevant to your camera model or determine them using other methods.
- main_path: Change to the main directory used by the other scripts.
- hidden_point_removal_factor: should be the same value as before.

_No other variables should need to be changed_

**This might take a long time to run!**

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

Finally, train the gaussian splatting! It is recommended to use these settings in sparse environments where you added a spherical background.

```
python train.py -s /path/to/main_path/ --eval

It is recommended to use these settings in sparse environments where you added a spherical background.

python train.py -s /path/to/main_path/ --eval --densify_until_iter 10000 --opacity_reset_interval 11000

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

You can then benchmark the visual reconstruction results with the following:
python render.py -m /path/to/main_path/
python metrics.py -m /path/to/main_path/

## Postprocessing:

If you want to filter out the spherical background after training or filter out most of the floating gaussians that might have been created during training, in `MESSER_for_3DGS/python_scripts/extra_utility_beta` run `post_process_gaussian_splat.py`. This filters out points reated during training that are not located close to the raw point cloud from fast-lio2. The following parameters can be changed:

- voxel_size: increase for faster but more rough filtering.
- filter_factor: the maximum distance to each voxel filtered points required to be kept in the point cloud. Increase for a larger radius around the voxels to be kept. Minimum 1.
- viz: keep true to vizualize the overlay and filtering.
