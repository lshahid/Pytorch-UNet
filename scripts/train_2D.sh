#!/bin/bash

# # Explore docker container in CHTC
# # Print working dir where I land when container starts
# echo -e "Working dir:\n$(pwd)\n"

# # Files in working dir
# echo -e "Files in working dir:\n$(ls -a)\n"

# # Print home directory of container
# echo -e "Home dir:\n$HOME\n"

# # Change to home directory
# cd ~
# echo -e "Working dir after changing to home dir: \n$(pwd)\n"
# echo -e "Files in home dir:\n$(ls -a)\n"

# # Change to workspace
# cd workspace/
# echo -e "Working dir after changing to workspace dir: \n$(pwd)\n"
# echo -e "Files in workspace dir:\n$(ls -a)\n"

# Previous shell script
# Change to autoseg_2D directory
#cd autoseg_2D/

# Untar data
tar xzf data.tar.gz

# Train the model
python3 /workspace/autoseg_2D/train.py --amp -w $(pwd) -e 2

# Remove the data directory
#rm -r data/

# Tar checkpoints
tar czf checkpoints.tar.gz checkpoints/
