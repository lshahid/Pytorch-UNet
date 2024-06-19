#!/bin/bash

# CPU and GPU information
lscpu
nvidia-smi

# Untar data
tar xzf data.tar.gz

# # Filter out images and masks with bladder
# python3 /workspace/autoseg_2D/scripts/mask_filter.py -w $(pwd) -t 500

# Train the model
time python3 /workspace/autoseg_2D/train.py --amp -w $(pwd) -e 30

# # Remove the data directories
# rm -r data/
# rm -r data_all/

# Tar checkpoints
tar czf checkpoints.tar.gz checkpoints/
