import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
import csv
import skimage as ski

im_1 = imread("/home/jonathan/dl_dataset/camera_01/image_01_1.jpg")
im_2 = imread("/home/jonathan/dl_dataset/camera_02/image_02_1.jpg")
with open('/home/jonathan/dl_dataset/rotations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        try:
            rots= np.vstack((rots,[row[0],row[1]]))
        except:
            rots = np.array([row[0],row[1]])

rot_s = rots[0,0]
rot_w = rots[0,1]

im_conc = np.hstack((im_1,im_2))

plt.imshow(im_conc)
plt.title(f"Nacelle rotation:{'% .2f'%rot_s} degrees. Wing rotation:{'% .2f'%rot_w} degrees")
plt.show()