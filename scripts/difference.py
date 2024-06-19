"""
This script generates the difference image
between true and predicted masks.
"""

import os
from PIL import Image
import numpy as np

# Masks directory
# Note: Make sure file names are identical in both directories
true_masks_dir = 'true_masks/'
pred_masks_dir = 'pred_masks/'
diff_masks_dir = 'diff_masks/'
true_masks_np_dir = 'true_masks_np/'
pred_masks_np_dir = 'pred_masks_np/'
diff_masks_np_dir = 'diff_masks_np/'

# Iterate over files in true masks directory
for mask_name in os.listdir(true_masks_dir):
    true_mask_name = os.path.join(true_masks_dir, mask_name)
    pred_mask_name = os.path.join(pred_masks_dir, mask_name)
    
    # Import masks
    true_mask = Image.open(true_mask_name).convert('L')
    pred_mask = Image.open(pred_mask_name).convert('L')
    
    # Convert to arrays
    true_arr = np.array(true_mask)/255
    pred_arr = np.array(pred_mask)/255
    
    # Difference masks
    diff_arr = true_arr - pred_arr
    diff_mask_name = os.path.join(diff_masks_dir, mask_name)

    # Save masks as numpy arrays
    true_mask_np_name = os.path.join(true_masks_np_dir, mask_name[:-4])
    pred_mask_np_name = os.path.join(pred_masks_np_dir, mask_name[:-4])
    diff_mask_np_name = os.path.join(diff_masks_np_dir, mask_name[:-4])
    np.save(true_mask_np_name, true_arr)
    np.save(pred_mask_np_name, pred_arr)
    np.save(diff_mask_np_name, diff_arr)
