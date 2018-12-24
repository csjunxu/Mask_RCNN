import os
import sys
import glob
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# from mrcnn.visualize import save_image # added by JX

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_0300.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
InIMAGE_DIR = "/home/csjunxu/Github/data/sate/train2018"
OutIMAGE_DIR = "/home/csjunxu/Github/data/sate/train2018Results"
if not os.path.isdir(OutIMAGE_DIR):
    os.makedirs(OutIMAGE_DIR)

## Configurations

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

## Create Model and Load Trained Weights

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask"])

## Class Names
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['Plane', 'Ships', 'Running_Track', 'Helicopter', 'Vehicles', 'Storage_Tanks', 'Tennis_Court',
               'Basketball_Court', 'Bridge', 'Roundabout', 'Soccer_Field', 'Swimming_Pool', 'baseball_diamond',
               'Buildings', 'Road', 'Tree', 'People', 'Hangar', 'Parking_Lot', 'Airport', 'Motorcycles', 'Flag',
               'Sports_Stadium', 'Rail_(for_train)', 'Satellite_Dish', 'Port', 'Telephone_Pole',
               'Intersection/Crossroads', 'Shipping_Container_Lot', 'Pier', 'Crane', 'Train', 'Tanks', 'Comms_Tower',
               'Cricket_Pitch', 'Submarine', 'Radar', 'Horse_Track', 'Hovercraft', 'Missiles', 'Artillery',
               'Racing_Track', 'Vehicle_Sheds', 'Fire_Station', 'Power_Station', 'Refinery', 'Mosques', 'Helipads',
               'Shipping_Containers', 'Runway', 'Prison', 'Market/Bazaar', 'Police_Station', 'Quarry', 'School',
               'Graveyard', 'Well', 'Rifle_Range', 'Farm', 'Train_Station', 'Crossing_Point', 'Telephone_Line',
               'Vehicle_Control_Point', 'Warehouse', 'Body_Of_water', 'Hospital', 'Playground', 'Solar_Panel']

## Run Object Detection
filenames = glob.glob(os.path.join(InIMAGE_DIR, "*.png"))
for counter, fl in enumerate(filenames):
    print("counter = {:5d}".format(counter))
    image_name = fl.split('/')[-1]
    output_path = os.path.join(OutIMAGE_DIR, image_name)
    if os.path.isfile(output_path):
        continue

    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(fl)
    # remove alpha channel
    image = image[:, :, :3]

    results = model.detect([image], verbose=1)

    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    fig = visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    # fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # print(fig.shape)
    plt.imsave(output_path, fig)
    # fig.imsave(output_path)


