import pycolmap as cm
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import csv
import os
from pyquaternion import Quaternion
from copy import deepcopy
from itertools import chain
import argparse
import collections
import struct

##################################################################
##################### DECLARE CONSTANTS ##########################
##################################################################
voxel_size = 0.2
min_x = 5 #min distance to keep points
max_x = 60 #max distance to keep points
f = 1108.5125019853992
h = 720
w = 1280
px = 640
py = 360
fov_x = 2*np.arctan2(w,(2*f))
fov_y = 2*np.arctan2(h,(2*f))
#projection matrix to project 3d points to image plane
proj_mat = np.array([[f, 0, px, 0],
                    [0, f, py, 0],
                    [0, 0, 1, 0]])

#rotation around y axis due to camera frame issue in IsaacSim
#transform_out = np.array([[-1.0, -0.0, 0.0, 0.0],   
#                          [0.0, 1.0, 0.0, 0.0],   
#                          [0.0, 0.0, -1.0, 0.0],   
#                          [0.0, 0.0, 0.0, 1.0]])
#transformation matrix to transform between world and camera frame
#trans_mat = np.linalg.pinv(np.array([[0.0, 0.0, 1.0, 0.0],
#                                     [-1.0, 0.0, 0.0, 0.0],
#                                     [0.0, -1.0, 0.0, 0.0],
#                                     [0.0, 0.0, 0.0, 1.0]]))

# Paths
main_path = '/home/jonathan/Reconstruction/windmill_stage'
image_path = os.path.join(main_path,'input')
pcd_path = os.path.join(main_path,'pcd')
reconstructed_path = os.path.join(main_path,'reconstructed.pcd')
transform_path = os.path.join(main_path,'transformations.csv')
output_path = os.path.join(main_path,'distorted/sparse/0')
cameras_txt_path = os.path.join(output_path,'cameras.txt')
cameras_bin_path = os.path.join(output_path,'cameras.bin')
images_txt_path = os.path.join(output_path,'images.txt')
images_bin_path = os.path.join(output_path,'images.bin')
points3D_txt_path = os.path.join(output_path,'points3D.txt')
points3D_bin_path = os.path.join(output_path,'points3D.bin')
if not os.path.isdir(output_path):
    os.makedirs(output_path)


##################################################################
##################################################################
##################################################################


##################################################################
########### FUNCTIONS FROM COLMAP read_write_model.py ############
##################################################################

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def write_images_text(images, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())
        ) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def write_points3D_text(points3D, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())
        ) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

##################################################################
##################################################################
##################################################################

##################################################################
####################### CUSTOM FUNCTIONS #########################
##################################################################

def convert_to_colmap_camera(cam_vec):

    cameras = {}
    camera_id = int(cam_vec[0])
    model = cam_vec[1]
    width = int(cam_vec[2])
    height = int(cam_vec[3])
    params = np.array(tuple(map(float, cam_vec[4:])))
    cameras[camera_id] = Camera(
        id=camera_id,
        model=model,
        width=width,
        height=height,
        params=params,
    )
    return cameras

def convert_to_colmap_images(im_l1,im_l2,N):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    for i in range(N):
        im_1 = im_l1[i]
        im_2 = im_l2[i]
        image_id = int(im_1[0])
        qvec = np.array(tuple(map(float, im_1[1:5])))
        tvec = np.array(tuple(map(float, im_1[5:8])))
        camera_id = int(im_1[8])
        image_name = im_1[9]
        xys = np.column_stack(
            [
                tuple(map(float, im_2[0::3])),
                tuple(map(float, im_2[1::3])),
            ]
        )
        point3D_ids = np.array(tuple(map(int, im_2[2::3])))
        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=xys,
            point3D_ids=point3D_ids,
        )
    return images

def convert_to_colmap_points3D(p3d,N):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    for i in range(N):
        p = p3d[i] 
        point3D_id = int(p[0])
        xyz = np.array(tuple(map(float, p[1:4])))
        rgb = np.array(tuple(map(int, p[4:7])))
        error = float(p[7])
        image_ids = np.array(tuple(map(int, p[8::2])))
        point2D_idxs = np.array(tuple(map(int, p[9::2])))
        points3D[point3D_id] = Point3D(
            id=point3D_id,
            xyz=xyz,
            rgb=rgb,
            error=error,
            image_ids=image_ids,
            point2D_idxs=point2D_idxs,
        )
    return points3D

##################################################################
##################################################################
##################################################################


##################################################################
######################### WRITE camera.txt #######################
##################################################################
#camera.txt
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] (f, cx, cy)
# Number of camera models: 1

camera_list = [1, "SIMPLE_PINHOLE", w, h, f, px, py]

cameras = convert_to_colmap_camera(camera_list)
write_cameras_text(cameras, cameras_txt_path)
write_cameras_binary(cameras, cameras_bin_path)
print("cameras.txt and cameras.bin created!")

##################################################################
##################################################################
##################################################################


##################################################################
######################### WRITE images.txt #######################
##################################################################
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2, mean observations per image: 2

#read pointcloud
pcd = o3d.io.read_point_cloud(reconstructed_path)
print("voxel downsampling of pointcloud...")
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
#mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
#o3d.visualization.draw_geometries([pcd,mesh_frame],zoom=0.3,
#                                  front=[0., 0., -1.],
#                                  lookat=[0., -2., 20],
#                                  up=[0., -1., 0.])
R,_ = np.shape(np.asarray(pcd.points))
point3d_id = np.linspace(0,R-1,R).astype(int)
print("Created 3D point indexing, found ",R,'points!')

#Determine number of images
k = 0
for file in os.listdir(image_path):
    k = k+1    
print('Found ', k,'images!')

#read transform table
with open(transform_path, 'r') as file:
    reader = csv.reader(file)
    transform_data = []
    for row in reader:
        row = [float(i) for i in row]
        transform_data.append(row)
transform_data = np.array(transform_data)
transform = np.reshape(transform_data,(-1,4,4))
N,_,_ = np.shape(transform)
print("Found",N,'transforms!')
im_l1 = {}
im_l2 = {}
print("Processing images...")
for i in range(N):
    if i%10==0:
        print(i,'/',N)
    q_transform = np.linalg.pinv(transform[i])
    #q_transform = np.matmul(transform[i],np.linalg.pinv(trans_mat))
    t = q_transform[:3,3]
    tx = t[0]
    ty = t[1]
    tz = t[2]
    q = rotmat2qvec(q_transform[:3,:3])
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    temp = deepcopy(pcd)
    temp.transform(q_transform)
    #o3d.visualization.draw_geometries([temp,mesh_frame],zoom=0.3,
    #                              front=[0., 0., -1.],
    #                              lookat=[0, -2., 20],
    #                              up=[0., -1., 0.])
    diameter = np.linalg.norm(np.asarray(temp.get_max_bound()) - np.asarray(temp.get_min_bound()))
    camera = [0, 0, 0]
    radius = diameter*8000
    _, pt_map = temp.hidden_point_removal(camera, radius)
    temp = temp.select_by_index(pt_map)
    
    points = np.asarray(temp.points)
    colors = np.asarray(temp.colors)
    temp_point3d_id = deepcopy(point3d_id)
    temp_point3d_id = np.squeeze(temp_point3d_id[pt_map])
    r_p,_ = np.shape(points)

    #Initial masking
    x = points[:,0] #3dimensional decomposition of lidar points
    y = points[:,1]
    z = points[:,2]

    #determining angle of each beam in pointcloud both horizontal and vertical angle
    theta_x = np.arctan2(x,z) #2d angles of points correlating to angles of image pixels
    theta_y = np.arctan2(-y,z)

    # filter out points outside the fov angles of the 
    ### Code for filtering points outside the fov of the camera
    positive_mask = np.where((np.abs(theta_x) < (fov_x/2)) & (np.abs(theta_y)<(fov_y/2)) & (z<max_x) & (z > min_x)) #filtering mask
    points = np.squeeze(points[positive_mask,:]) #filtered points
    colors = np.squeeze(colors[positive_mask,:])
    temp_point3d_id = np.squeeze(temp_point3d_id[positive_mask])
    ### Code for projecting the point cloud onto the image plane ###
    r_new,_ = np.shape(points) #shape of filtered array
    extend_homogenous = np.ones((r_new,1)) #creating homogenous extender (r_new,1)

    points_homogenous = np.hstack((points,extend_homogenous)) #homogenous array (r_new,4)

    points_proj = points_homogenous@proj_mat.T #initial projection of points to image plane
    x = np.divide(points_proj[:,0],points_proj[:,2]) #normalizing to fit with image size
    y = np.divide(points_proj[:,1],points_proj[:,2])
    z = extend_homogenous
    points_out = np.column_stack((x,y,z)) # extending array to 3d to visualize as pointcloud (not necessary)
    positive_mask = np.where((points_out[:,0] < w) & (points_out[:,1] < h) & (points_out[:,0] > 0) &(points_out[:,1] > 0) ) #filtering mask

    points_out = np.squeeze(points_out[positive_mask,:])
    colors_out = np.squeeze(colors[positive_mask,:])
    temp_point3d_id = np.squeeze(temp_point3d_id[positive_mask])
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    #create line 1 of image_N in image.txt
    im_l1[i] = [i+1,qw,qx,qy,qz,tx,ty,tz,1,"img_"+str(i)+".jpg"]
    im_l2[i] = list(chain.from_iterable(zip(points_out[:,0],points_out[:,1],temp_point3d_id.astype(int))))
images = convert_to_colmap_images(im_l1,im_l2,N)

write_images_text(images, images_txt_path)
write_images_binary(images, images_bin_path)
print("images.txt and images.bin created!")

##################################################################
##################################################################
##################################################################


##################################################################
######################### WRITE points3D.txt #####################
##################################################################
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 3, mean track length: 3.3334
p3d = {}
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
p_x = points[:,0]
p_y = points[:,1]
p_z = points[:,2]
c_r = colors[:,0]
c_g = colors[:,1]
c_b = colors[:,2]
error = 1.0
track = [0, 0]
print('Processing Points...')
for i in range(R):
    id = point3d_id[i]
    p3d[i] = [point3d_id[i], p_x[i],p_y[i],p_z[i],c_r[i],c_g[i],c_b[i], error, 0, 0 ]
    
points3D = convert_to_colmap_points3D(p3d,R)
write_points3D_text(points3D, points3D_txt_path)
write_points3D_binary(points3D, points3D_bin_path)
print("points3D.txt and points3D.bin created!")