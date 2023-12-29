"""
This script generates the difference image
between true and predicted masks.
"""

import os
from PIL import Image
import numpy as np
import imageio

# Masks directory
# Note: Make sure file names are identical in both directories
true_masks_dir = '../output/true_masks/'
pred_masks_dir = '../output/pred_masks/'
diff_masks_dir = '../output/diff_masks/'

# Iterate over files in true masks directory
for mask_name in os.listdir(true_masks_dir):
    true_mask_name = os.path.join(true_masks_dir, mask_name)
    pred_mask_name = os.path.join(pred_masks_dir, mask_name)
    
    # Import masks
    true_mask = Image.open(true_mask_name).convert('L')
    pred_mask = Image.open(pred_mask_name).convert('L')
    
    # Convert to arrays
    true_arr = np.array(true_mask)
    pred_arr = ((np.array(pred_mask) - 255) * -1).astype('uint8')
    
    # Difference masks
    diff_arr = pred_arr - true_arr
    diff_mask_name = os.path.join(diff_masks_dir, mask_name)
    imageio.imsave(diff_mask_name, diff_arr)
    
# man_img = Image.open("test3/output3/truth/BPH009_Phase023_0057.gif").convert('L')
# man_arr = np.array(man_img)

# auto_img = Image.open("test3/output3/BPH009_Phase023_0057.gif").convert('L')
# auto_arr = (np.array(auto_img) - 255) * -1
# auto_arr = auto_arr.astype('uint8')

# diff_arr = auto_arr - man_arr
# print(np.max(diff_arr))
# print(np.min(diff_arr))

# imageio.imsave('BPH009_Phase023_0057_diff.gif', diff_arr)
