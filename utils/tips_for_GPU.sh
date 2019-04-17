#!/bin/bash
###############################################################################
#
# Author: hoho2b
# Created Time: 2019.04.16 21:30:15
#
###############################################################################


# https://github.com/matterport/Mask_RCNN/issues/958
# > AttributeError: 'ParallelModel' object has no attribute '_is_graph_network'
conda install keras=2.1.3 tensorflow-gpu

# https://github.com/matterport/Mask_RCNN/pull/1082
sed -i'' 's!\[self.config.BATCH_SIZE![self.config.IMAGES_PER_GPU!' ./Mask_RCNN/mrcnn/model.py
