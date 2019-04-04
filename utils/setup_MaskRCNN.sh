#!/bin/bash
###############################################################################
# 
# Author: hoho2b
# Created Time: 2019.04.04 01:01:15
# 
###############################################################################

# clone Mask_RCNN repo
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN

# python env checkout by `requirements.txt`
pip install -r requirements.txt 

# Install COCOAPI
pip install pycocotools
# pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

# (Automatic after v2.1) Download pre-trained COCO weights 
wget -c https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

###############################################################################



