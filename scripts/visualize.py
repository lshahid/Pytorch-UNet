"""
This script visualizes the images with their
corresponding masks and saves the figures.
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Directories
pred_imgs_dir = 'pred_imgs/'
true_masks_np_dir = 'true_masks_np/'
pred_masks_np_dir = 'pred_masks_np/'
diff_masks_np_dir = 'diff_masks_np/'
imgs_dir = 'imgs/'

# Pattern to match all files in the images directory
pattern = os.path.join(pred_imgs_dir, '*')

# Iterate over files in images directory
for img_path in glob.glob(pattern):
    img_name = os.path.basename(img_path)  # Extract the file name from the path
    true_mask_name = os.path.join(true_masks_np_dir, img_name[:-4] + '.npy')
    pred_mask_name = os.path.join(pred_masks_np_dir, img_name[:-4] + '.npy')
    diff_mask_name = os.path.join(diff_masks_np_dir, img_name[:-4] + '.npy')

    # Import image and masks
    img = plt.imread(img_path)
    true_mask = np.load(true_mask_name)
    pred_mask = np.load(pred_mask_name)
    diff_mask = np.load(diff_mask_name)

    # Visualization
    plt.figure(figsize=(15, 3))

    plt.subplot(1, 4, 1)
    plt.title('Image')
    plt.imshow(img)
    plt.xticks([])  # Hide x-axis numbers
    plt.yticks([])  # Hide y-axis numbers

    plt.subplot(1, 4, 2)
    plt.title('True Mask')
    plt.imshow(true_mask, cmap='gray')
    plt.xticks([])  # Hide x-axis numbers
    plt.yticks([])  # Hide y-axis numbers

    plt.subplot(1, 4, 3)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask, cmap='gray')
    plt.xticks([])  # Hide x-axis numbers
    plt.yticks([])  # Hide y-axis numbers

    plt.subplot(1, 4, 4)
    plt.title('Difference Mask')
    plt.imshow(diff_mask, cmap='gray')
    plt.xticks([])  # Hide x-axis numbers
    plt.yticks([])  # Hide y-axis numbers

    # Save the figure
    save_path = os.path.join(imgs_dir, img_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
