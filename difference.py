import os
from PIL import Image
import numpy as np

# Masks directory
# Note: Make sure file names are identical in both directories
true_masks_dir = 'true_masks/'
pred_masks_dir = 'pred_masks/'
diff_masks_dir = 'diff_masks/'
#true_masks_np_dir = 'true_masks_np/'
#pred_masks_np_dir = 'pred_masks_np/'
#diff_masks_np_dir = 'diff_masks_np/'

# Dice score
def DICE_Coeff(true_mask_arr, pred_mask_arr):
    #Original = Image.open(original_mask_path)
    #Predicted = Image.open(predicted_mask_path)
    epsilon = 1e-6

    #Original = np.array(Original)
    #Predicted = np.array(Predicted)
    if np.count_nonzero(true_mask_arr)!=0:
        true_mask_arr = np.where(true_mask_arr == 0, 1, 0)
    #Predicted = Predicted / 255
    Intersection = np.sum(true_mask_arr * pred_mask_arr)
    DICE = (2. * Intersection + epsilon) / (np.sum(true_mask_arr) + np.sum(pred_mask_arr) + epsilon)
    return DICE

DICE_Dict = {}

# Iterate over files in true masks directory
for mask_name in sorted(os.listdir(true_masks_dir)):
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
    diff_mask_name = os.path.join(diff_masks_dir, mask_name.split('.')[0])

    # Calculate dice score
    DICE_Score = DICE_Coeff(true_arr, pred_arr)
    Name = mask_name.split('.')[0]

    DICE_Dict[Name] = DICE_Score
    # Save difference mask as numpy array
    np.save(diff_mask_name, diff_arr)

DICE_Max_Name = max(DICE_Dict, key=DICE_Dict.get)
DICE_Max = DICE_Dict[DICE_Max_Name]
DICE_Min_Name = min(DICE_Dict, key=DICE_Dict.get)
DICE_Min = DICE_Dict[DICE_Min_Name]
