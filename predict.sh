#!/bin/bash

# CPU and GPU information
lscpu
nvidia-smi

# Untar images to segment
tar xzf pred_imgs.tar.gz

# Predicted mask directory
mkdir pred_masks

# Segment images
time python3 predict.py -m checkpoint_epochXX.pth -i pred_imgs/ -o pred_masks/ -t 0.5 -c 1 -s 1

# Tar predicted masks
tar czf pred_masks.tar.gz pred_masks/

# Remove data
rm -r pred_imgs/ pred_imgs.tar.gz
