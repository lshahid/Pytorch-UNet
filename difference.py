import os
from PIL import Image
import numpy as np
import pdb

# Masks directory
# Note: Make sure file names are identical in both true and predicted masks directories
true_masks_dir = 'true_masks/'
pred_masks_dir = 'pred_masks/'
diff_masks_dir = 'diff_masks/'

# Dice score
def DICE_Coeff(true_mask_arr, pred_mask_arr):
    #Original = Image.open(original_mask_path)
    #Predicted = Image.open(predicted_mask_path)
    epsilon = 1e-6

    #Original = np.array(Original)
    #Predicted = np.array(Predicted)
    # if np.count_nonzero(true_mask_arr)!=0:
    #     true_mask_arr = np.where(true_mask_arr == 0, 1, 0)
    #Predicted = Predicted / 255
    Intersection = np.sum(true_mask_arr * pred_mask_arr)
    DICE = (2. * Intersection + epsilon) / (np.sum(true_mask_arr) + np.sum(pred_mask_arr) + epsilon)
    return DICE

def dice(im1, im2, epsilon = 1e-6):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() + epsilon) / ((im1.sum() + im2.sum()) + epsilon)

dice_dict = {}

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
    
    # # Difference masks
    # diff_arr = true_arr - pred_arr
    
    # # Save difference mask as numpy array
    # diff_mask_name = os.path.join(diff_masks_dir, mask_name.split('.')[0])
    # np.save(diff_mask_name, diff_arr)
    
    # Calculate dice score
    DICE_Score = DICE_Coeff(true_arr, pred_arr)
    Name = mask_name.split('.')[0]
    dice_dict[Name] = [dice(true_arr, pred_arr), DICE_Coeff(true_arr, pred_arr)]
    pdb.set_trace()

pdb.set_trace()

#DICE_Max_Name = max(DICE_Dict, key=DICE_Dict.get)
#DICE_Max = DICE_Dict[DICE_Max_Name]
#DICE_Min_Name = min(DICE_Dict, key=DICE_Dict.get)
#DICE_Min = DICE_Dict[DICE_Min_Name]
