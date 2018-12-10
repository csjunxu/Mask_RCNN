import os
import matplotlib.pyplot as plt
import numpy as np
import glob
# height, width
# (37888, 66560)
root_dir = "sliced_mask_JX"

file_list = glob.glob(os.path.join(root_dir, "*.png"))

basename_list = [fl.split('/')[-1].split('.')[0].split('_') for fl in file_list]

x_list = []
y_list = []
for fl in basename_list:
    x_list.extend(fl[:2])
    y_list.extend(fl[2:])


x_list = [int(x) for x in x_list]
y_list = [int(y) for y in y_list]

height = max(x_list)
width  = max(y_list)
print("total image size is ", height, width, 3)

total_image = np.zeros((height, width, 4))
for counter, fl in enumerate(file_list):
    print("counter = {:5d}".format(counter))
    box_location = fl.split('/')[-1].split('.')[0].split('_')
    print(box_location)
    x_min, x_max, y_min, y_max = int(box_location[0]), int(box_location[1]), \
        int(box_location[2]), int(box_location[3])

    img = plt.imread(fl)
    total_image[x_min:x_max, y_min:y_max, :] = img

plt.imsave("masked_output.png", total_image)


