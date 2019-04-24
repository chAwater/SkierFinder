#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Science
import numpy  as np
import pandas as pd
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Tricks
sns.set(style='ticks', context='talk', font_scale=1.15)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os, sys
import skimage.io

from glob import glob

# Root directory of the project
ROOT_DIR = os.path.abspath("/mnt/gpfs/Users/chenhe/Playground/ski/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize


# In[3]:


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

vip_class = ['person','skis','snowboard']


# In[4]:


from utils.Tools import extInBoxPixels,squareBox,Show_Img,IMAGE_SHAPE

from multiprocessing import Pool

def mapper(row_it):
    (idx, row) = row_it
    ext_Box = extInBoxPixels(row, getMask=True)
    # Wrong size images, return a placeholder: a white board
    if ext_Box is None:
        return np.zeros((150,150,3),dtype=int)+255
    else:
        sqr_Box = squareBox(ext_Box)
        return sqr_Box


# In[5]:


from keras.applications.vgg19    import preprocess_input as pre1, VGG19 
from keras.applications.resnet50 import preprocess_input as pre2, ResNet50
from keras.applications.densenet import preprocess_input as pre3, DenseNet201

model1 = VGG19(
    weights = 'imagenet', 
    pooling = 'max',
    input_shape = (150,150,3),
    include_top = False
)
model2 = ResNet50(
    weights = 'imagenet', 
    pooling = 'max',
    input_shape = (150,150,3),
    include_top = False
)
model3 = DenseNet201(
    weights = 'imagenet', 
    pooling = 'max',
    input_shape = (150,150,3),
    include_top = False
)


# ---

# # Load data

# In[6]:


data_it  = pd.read_csv('wanlong.csv.gz',chunksize=10000)
clean_df = pd.read_pickle('bulk_clean_df.pkl')

def df_chunk_to_boxIN(df, pool):
    df_IN = pd.concat(
        [
            df
                .drop(['imgID','masks'],axis=1)
                .apply(lambda col: pd.to_numeric(col,errors='coerce'), axis=1),
            df['masks'],
            clean_df['box_size'],
            clean_df['imgID'],
        ], 
        axis=1,
        join='inner',
    )

    df_IN['imgID'] = (
        './huabei/wanlong/' + 
        (
            pd.to_datetime(
                df_IN['ref_day'].astype(int), unit='D', origin=pd.Timestamp('2019-03-01')
            )
            .apply( lambda x: x.strftime('%Y%m%d') )
        ) +
        '/' + df_IN['imgID'] + 'jpeg'

    )

    box_IN = pool.map( mapper, df_IN.query('(class_ids==1 & box_size>=1 & scores>=0.9)').iterrows() )
    box_IN = np.stack(box_IN)
    return box_IN


# In[7]:


pool = Pool(32)

for idx, df in enumerate(data_it):
    box_IN = df_chunk_to_boxIN(df, pool)

    v1 = model1.predict( pre1(box_IN) )
    v2 = model2.predict( pre2(box_IN) )
    v3 = model3.predict( pre3(box_IN) )

    np.save('v1.'+str(idx),v1)
    np.save('v2.'+str(idx),v2)
    np.save('v3.'+str(idx),v3)

pool.close()
pool.join()


# In[8]:


files =  sorted(
    glob('v1.*.npy'),
    key = lambda x: int( os.path.basename(x)[3:-4] )
)

v_vgg19 = []

for file in files:
    v = np.load(file)
    v_vgg19.append(v)
    
v_vgg19 = np.concatenate(v_vgg19)
np.save('v_vgg19',v_vgg19)


# In[9]:


files =  sorted(
    glob('v2.*.npy'),
    key = lambda x: int( os.path.basename(x)[3:-4] )
)

v_resnet = []

for file in files:
    v = np.load(file)
    v_resnet.append(v)
    
v_resnet = np.concatenate(v_resnet)
np.save('v_resnet',v_resnet)


# In[10]:


files =  sorted(
    glob('v3.*.npy'),
    key = lambda x: int( os.path.basename(x)[3:-4] )
)

v_densenet = []

for file in files:
    v = np.load(file)
    v_densenet.append(v)
    
v_densenet = np.concatenate(v_densenet)
np.save('v_densenet',v_densenet)


# In[ ]:




