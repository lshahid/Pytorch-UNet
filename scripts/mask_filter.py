"""
This script filters out image-mask pairs on the sides of the
field of view that do not have a bladder. This ensures that
images with bladders are used for training.
"""

import argparse
from pathlib import Path
import os
import imageio
import numpy as np
import shutil

def filter(work_dir: str='..', thresh: int=500):

    # Concatenate working directory to paths
    dir_img_all = Path(work_dir) / 'data_all/imgs/'
    dir_masks_all = Path(work_dir) / 'data_all/masks/'
    dir_img_filtered = Path(work_dir) / 'data/imgs/'
    dir_masks_filtered = Path(work_dir) / 'data/masks/'

    # Iterate over all masks
    print('List of images used for training')

    for subject in os.listdir(dir_masks_all):
        for mask_name in os.listdir(os.path.join(dir_masks_all, subject)):

            # Import mask
            mask = imageio.imread(os.path.join(dir_masks_all, subject, mask_name))
            mask_arr = np.array(mask)

            # Number of bladder pixels
            num_pixel = np.count_nonzero(mask_arr)

            if num_pixel >= thresh:

                # Make directory for images and masks with bladder
                Path(os.path.join(dir_img_filtered, subject)).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(dir_masks_filtered, subject)).mkdir(parents=True, exist_ok=True)

                # Move images and masks
                shutil.move(os.path.join(dir_img_all, subject, mask_name[:-4] + '.jpg'),
                            os.path.join(dir_img_filtered, mask_name[:-4] + '.jpg'))
                shutil.move(os.path.join(dir_masks_all, subject, mask_name),
                            os.path.join(dir_masks_filtered, mask_name))

                print(mask_name[:-4])

    print('\n')

# Command-line arguments
def get_args():
    parser = argparse.ArgumentParser(description='Filter out image-mask pairs that have a bladder.')
    parser.add_argument('--work-dir', '-w', dest='wd', type=str, default='..', help='Working directory')
    parser.add_argument('--threshold', '-t', type=int, default=500, help='Threshold for number of pixels')

    return parser.parse_args()

# Get arguments, and filter out images and masks
args = get_args()
filter(work_dir=args.wd, thresh=args.threshold)
