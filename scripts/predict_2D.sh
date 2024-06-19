#!/bin/bash

# CPU and GPU information
lscpu
nvidia-smi

# Untar images to segment
tar xzf pred_imgs.tar.gz

# Predicted mask directory
mkdir pred_masks

# Segment images
time python3 /workspace/autoseg_2D/mpredict.py -m checkpoint.pth -id pred_imgs/ -od pred_masks/

# Tar predicted masks
tar czf pred_masks.tar.gz pred_masks/
