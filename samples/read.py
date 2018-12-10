from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import sys
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

img = Image.open("/home/csjunxu/Project/20jpng.png")
img = np.array(img, dtype=np.uint8)
width_step  = 1024
height_step = 1024
height_number = img.size[0] // height_step
width_number  = img.size[1] // width_step
print(img.size)
# img = cv2.imread("/home/csjunxu/Project/20jpng.png")
# img = plt.imread("/home/csjunxu/Project/20jpng.png")
# img = np.random.random((3865, 6656, 4))
# height_number = img.shape[0] // height_step
# width_number = img.shape[1] // width_step
# print(img.shape)




print(height_number, width_number)
output_dir = "/home/csjunxu/Project/sliced_img_JX"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for i in range(height_number):
    for j in range(width_number):
        img_i_j = img.crop((i*height_step, j*width_step, (i+1)*height_step, (j+1)*width_step))
        # img_i_j = img[i*height_step:(i+1)*height_step, j*width_step:(j+1)*width_step, :]
        outfile = os.path.join(output_dir, str(i*height_step)\
                                    +'_' + str((i+1)*height_step)\
                                    +'_' +str(j*width_step)\
                                    +'_'+str((j+1)*width_step)\
                                    +'.png')
        plt.imsave(outfile, img_i_j)
        print(i+1,'/',height_number,';',j+1,'/',width_number,'is done!')

