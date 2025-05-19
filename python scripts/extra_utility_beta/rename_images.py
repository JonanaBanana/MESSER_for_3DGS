import cv2
import numpy as np
import os

main_path = '/home/jonathan/Reconstruction/outdoor_windmill_custom'
img_path = os.path.join(main_path,'input')
output_path = os.path.join(main_path,'images')
if not os.path.isdir(output_path):
    os.makedirs(output_path)
#Determine number of images
k = 0
for file in os.listdir(img_path):
    k = k+1    
print('Found ', k,'images!')

for i in range(k):
    img = cv2.imread(img_path+'/img_'+str(i)+'.jpg')
    if i >= 10:
        im_string = '/img_000' + str(i) + '.jpg'
        if i >= 100:
            im_string = '/img_00' + str(i) + '.jpg'
            if i >= 1000:
                im_string = '/img_0' + str(i) + '.jpg'
                if i >= 10000:
                    im_string = '/img_' + str(i) + '.jpg'
    else:
        im_string = '/img_0000' + str(i) + '.jpg'
    cv2.imwrite(output_path + im_string,img)