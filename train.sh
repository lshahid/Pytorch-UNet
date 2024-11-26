#!/bin/bash

# CPU and GPU information
lscpu
nvidia-smi

# Untar data
tar xzf data_1c_s.tar.gz

# Train the model
time python3 train.py --amp -e 30

# Tar checkpoints
tar czf checkpoints.tar.gz checkpoints/

# Remove data
rm -r data_1c_s/ data_1c_s.tar.gz
