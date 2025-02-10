import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
global pcd_conc

def rgbd_to_pcd(count,data):
    global pcd_conc

    source_color = o3d.io.read_image('/home/jonathan/Reconstruction_Images/rgbd_folder/rgb/rgb_%d.png'%count)
    source_depth = o3d.io.read_image('/home/jonathan/Reconstruction_Images/rgbd_folder/depth/depth_%d.png'%count)

    #loading camera intrinsics
    K = np.array([[1108.5125019853992, 0.0, 640.0],
                  [0.0, 1108.5125019853992, 360.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = K

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth, depth_scale=0.0041, convert_rgb_to_intensity=False, depth_trunc=100)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)
    #np_points = np.asarray(pcd.points)
    #p_x = np_points[:,2]
    #p_y = np_points[:,0]
    #p_z = -1*np_points[:,1]
    #np_points = np.vstack((p_x,p_y,p_z)).T
    #pcd.points = o3d.utility.Vector3dVector(np_points)
    transform = data[(i)*4:(i+1)*4]
    print(transform)
    pcd.transform(transform)
    pcd_ds = pcd.uniform_down_sample(50)
    if i==0:
        pcd_conc = pcd_ds
        vis.add_geometry(pcd_conc)
        
    else:
        pcd_conc.points.extend(pcd_ds.points)
        pcd_conc.colors.extend(pcd_ds.colors)
        vis.update_geometry(pcd_conc)
    vis.poll_events()
    vis.update_renderer()
    o3d.io.write_point_cloud('/home/jonathan/Reconstruction_Images/rgbd_folder/pcds/pcl_%d.pcd'%count, pcd)
    

if __name__ == '__main__':
    with open('/home/jonathan/Reconstruction_Images/rgbd_folder/transformations.csv', 'r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            row = [float(i) for i in row]
            data.append(row)
    
    data = np.array(data)
    total_count = int(len(data[:,0])/4)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    for i in range(0,total_count):
        rgbd_to_pcd(i,data)
    o3d.visualization.draw_geometries([pcd_conc,mesh_frame])
