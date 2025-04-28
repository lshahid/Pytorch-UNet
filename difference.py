import argparse
import os
from PIL import Image
import numpy as np
import statistics
from hausdorff_distance_2D import hausdorff_distance_mask

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

def get_args():
    parser = argparse.ArgumentParser(description='Difference between true and predicted masks')
    parser.add_argument('--true-dir', '-i', metavar='TRUE_DIR', help='True mask directory', required=True)
    parser.add_argument('--pred-dir', '-o', metavar='PRED_DIR', help='Predicted mask directory', required=True)
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the difference masks')
    parser.add_argument('--diff-dir', '-d', metavar='DIFF_DIR', help='Difference mask directory')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    true_masks_dir = args.true_dir
    pred_masks_dir = args.pred_dir
    
    dice_dict = {}
    HD_dict = {}

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
        
        # Calculate dice score
        DICE_Score = DICE_Coeff(true_arr, pred_arr)
        Name = mask_name.split('.')[0]
        dice_dict[Name] = DICE_Score
        
        #Calculate HD Distance
        if (len(np.unique(true_arr))>1 and len(np.unique(pred_arr))>1): 
            HD_Distance = hausdorff_distance_mask(true_arr,pred_arr,'standard')
            HD_dict[Name] = HD_Distance
            
        # Difference masks
        if not args.no_save:
            diff_arr = true_arr - pred_arr
            
            # Save difference mask as numpy array
            diff_masks_dir = args.diff_dir
            diff_mask_name = os.path.join(diff_masks_dir, mask_name.split('.')[0])
            np.save(diff_mask_name, diff_arr)

    dice_scores = list(dice_dict.values())
    HD_distances = list(HD_dict.values())

    # Print Dice score and Hausdorff distance
    print('2D DICE Score Statistics\n'
          f"Mean: {statistics.mean(dice_scores)}\n"
          f"Min: {np.min(dice_scores)}\n"
          f"1st quartile: {statistics.quantiles(dice_scores, n=4)[0].item()}\n"
          f"Median: {statistics.quantiles(dice_scores, n=4)[1].item()}\n"
          f"3rd quartile: {statistics.quantiles(dice_scores, n=4)[2].item()}\n"
          f"Max: {np.max(dice_scores)}")
    
    print('\n2D Hausdorff Distance Statistics\n'
          f"Mean: {statistics.mean(HD_distances)}\n"
          f"Min: {np.min(HD_distances)}\n"
          f"1st quartile: {statistics.quantiles(HD_distances, n=4)[0].item()}\n"
          f"Median: {statistics.quantiles(HD_distances, n=4)[1].item()}\n"
          f"3rd quartile: {statistics.quantiles(HD_distances, n=4)[2].item()}\n"
          f"Max: {np.max(HD_distances)}")
