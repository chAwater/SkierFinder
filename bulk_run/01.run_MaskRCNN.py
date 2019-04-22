#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[ ]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath('/path_to/Mask_RCNN/')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[ ]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT        = 2
    IMAGES_PER_GPU   = 6
    MAX_GT_INSTANCES = 50

config = InferenceConfig()
# config.display()


# ## Create Model and Load Trained Weights

# In[ ]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[ ]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection

# In[ ]:


import pandas as pd

from glob            import glob
from multiprocessing import Pool

IMAGE_SHAPE = (467,700,3)
BATCH_SIZE  = config.GPU_COUNT * config.IMAGES_PER_GPU


# In[ ]:


# ImagesArray Generator
def files2images(names, pool):
    total = len(names)
    for idx in range(0, total, BATCH_SIZE):
        end_idx = idx + BATCH_SIZE
        if end_idx <= total:
            # Enough images in this batch
            ImagesArray = pool.map(skimage.io.imread, names[idx: end_idx])
            yield (idx, ImagesArray)
#             yield (idx, list( map(skimage.io.imread, names[idx: end_idx]) ))
        else:
            # Fill empty images if it is not enough
            n_fill = end_idx - total
            img_empty = np.zeros( IMAGE_SHAPE )
            yield (idx, pool.map(skimage.io.imread, names[idx:   total]) + [img_empty]*n_fill)
#             yield (idx, list( map(skimage.io.imread, names[idx:   total]) ) + [img_empty]*n_fill)

# Formatting MaskRCNN result
def resultDict2df(zip_input):
    (file_name, r) = zip_input

    result_df = pd.DataFrame(
        np.stack(
            (
                r['scores'],
                r['class_ids']
            )
        ).T ,columns=['scores','class_ids']
    )

    try:
        file_prefix = os.path.basename(file_name)[:-4]
        result_df['imgID'] = file_prefix
    except:
        return None

    n = result_df.shape[0]
    if n == 0:
        result_df = result_df.astype( {'class_ids':np.int16, 'scores':np.float32} )
        return result_df

    result_df[['x1','y1','x2','y2']] = pd.DataFrame(r['rois'].tolist())
    result_df['masks'] = list(
        map(
            lambda x: ''.join(
                list(
                    map(
                        lambda b: str( int(b) ),
                        x
                    )
                ) 
            ),
            np.rollaxis(r['masks'], 2, 0).reshape(n,-1)
        )
    )

    result_df = result_df.astype(
        {
            'scores'    : np.float32,
            'class_ids' : np.int16,
            'x1'        : np.int16,
            'x2'        : np.int16,
            'y1'        : np.int16,
            'y2'        : np.int16, 
        }
    )
    return result_df


# ---

# In[ ]:


# ============================================================================ #
# ===============================  Focus HERE  =============================== #
# ============================================================================ #


# ---

# In[ ]:


# file_names =  sorted(
#     glob("./imgs_pw_20190331_wl_n1/*.jpg"),
#     key = lambda x: int( os.path.basename(x)[:-4] )
# )

days =  sorted(
    glob('./huabei/wanlong/*'),
    key = lambda x: int( os.path.basename(x) )
)


# In[ ]:


pool = Pool()

for day in days:
    print ( 'Processing: ' + os.path.basename(day) )
    file_names = glob( os.path.join(day,'*.jpeg') )

    try:
        # Generate ImagesArray in batch
        for idx, g in files2images(file_names, pool):
            # Run MaskRCNN
            results = model.detect(g, verbose=0)
            # Get file names
            img_IDs = file_names[idx:idx+BATCH_SIZE]

            # Formatting result
            # Using zip to prevent out of range
            df_results = pool.map(resultDict2df, zip(img_IDs,results) )

            # Save dataframe
            df_results = pd.concat(df_results, ignore_index=True, sort=False)
            df_results.to_pickle( os.path.join(day, str(idx)+'.pkl') )
    except:
        print ( '@'+os.path.basename(day)+', Something wrong ...' )
        continue
    
pool.close()
pool.join()


# In[ ]:





# In[ ]:




