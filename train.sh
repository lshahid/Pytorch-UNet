#!/bin/bash

# CPU and GPU information
lscpu
nvidia-smi

# Untar data
tar xzf data.tar.gz

# Train the model
time python3 train.py --amp -e 30 -c 1 -t 0.5 -s 1

# Tar checkpoints
tar czf checkpoints.tar.gz checkpoints/

# Remove data
rm -r data/ data.tar.gz
