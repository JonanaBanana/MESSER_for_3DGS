import numpy as np
import open3d as o3d
import csv
import os
from copy import deepcopy

########################## PATHS ################################
file_path = os.path.dirname(__file__)  
main_path = os.path.join(file_path, '../example_stage_warehouse')
downsampled_path = os.path.join(main_path,'downsampled_point_cloud.pcd')
csv_path = os.path.join(main_path,'point_cloud_color_information.csv')
out_path = os.path.join(main_path,'reconstructed.pcd')
#################################################################

def main(args=None):
    pcd_fl = o3d.io.read_point_cloud(downsampled_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = pcd_fl.points
    points = np.asarray(pcd.points)
            
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        data_list = []
        for row in reader:
            row = [float(i) for i in row]
            data_list.append(row)
    data_list = np.array(data_list)
    data_list = np.reshape(data_list,(-1,4))
    idx_list = data_list[:,0].astype(int)
    sorter_idx = np.argsort(idx_list)
    idx_list = idx_list[sorter_idx]
    L = np.squeeze(np.shape(idx_list))
    colors_list = data_list[:,1:].astype(float)
    colors_list = colors_list[sorter_idx,:]
    R,_ = np.shape(np.asarray(pcd.points))
    visible_idx = np.sort(np.unique(idx_list))
    M = np.shape(np.asarray(visible_idx))[0]
    points_out = points[visible_idx,:]
    colors_out = np.zeros((R,3))

    print('Total input data points: ',L)
    print("Found ",R,'points in point cloud')
    print("Visible points from images:",M)
    print("Generating color for each point")
    i = 0

    for idx in visible_idx:
            flag = 1
            colors = np.empty((0,3))
            while flag==1:
                if i<L:
                    if idx_list[i] == idx:
                        colors = np.append(colors,np.expand_dims(colors_list[i,:],axis=0),axis=0)
                        i = i+1
                        if i%100000 == 0:
                            print("Progress: "+str(i)+"/"+str(L))
                    else:
                        colors = np.sort(colors, axis=0, kind='mergesort')
                        colors = np.median(colors,axis=0)
                        colors_out[idx,:] = colors
                        flag=0
                else:
                    break

    colors_out = colors_out[visible_idx,:]
    pcd.points = o3d.utility.Vector3dVector(points_out)
    pcd.colors = o3d.utility.Vector3dVector(colors_out)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd,mesh_frame],zoom=0.3,
                                    front=[0., 0., -1.],
                                    lookat=[0., -2., 20],
                                    up=[0., -1., 0.])

if __name__ == '__main__':
    main()