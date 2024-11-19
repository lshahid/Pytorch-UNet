#!/bin/bash

# CPU and GPU information
lscpu
nvidia-smi

# Untar data
tar xzf data_1c_ns.tar.gz

# Train the model
time python3 train.py --amp -e 10

# Tar checkpoints
tar czf checkpoints_1c_ns.tar.gz checkpoints_1c_ns/

# Remove data
rm -r data_1c_ns/ data_1c_ns.tar.gz
