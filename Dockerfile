# Use Nvidia PyTorch with Python 3 as base image
FROM nvcr.io/nvidia/pytorch:23.07-py3

# Remove all existing files and set working directory
RUN rm -rf /workspace/*
WORKDIR /workspace/autoseg_2D

# Install packages listed in requirements.txt
ADD requirements.txt .
RUN pip3 install --no-cache-dir --upgrade --pre pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Add all files in Github repository
ADD . .

# Reset working directory
WORKDIR /workspace
