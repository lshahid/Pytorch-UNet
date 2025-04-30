import argparse
import os, pdb
from PIL import Image
import numpy as np
import pandas as pd
import shutil
from scipy import ndimage

#Function to check for isolated pixels within the time phase
def check_isolation(df_time_phase, i, thr):
    slice_info = df_time_phase.iloc[i]
    num_ones_i = slice_info['Num_Ones']        

    # If current slice is smaller than the threshold
    if (num_ones_i <= thr and num_ones_i > 0):
        # If current slice is on the edges then it should be empty
        if (i == 0 or i == 1 or i == (len(df_time_phase)-2) or i == (len(df_time_phase)-1)):
            Keep = False

        # If current slice is not on the edges
        else:
            #num_ones_im2 = df_time_phase.iloc[i-2]['Num_Ones']
            num_ones_im1 = df_time_phase.iloc[i-1]['Num_Ones']
            num_ones_ip1 = df_time_phase.iloc[i+1]['Num_Ones']
            #num_ones_ip2 = df_time_phase.iloc[i+2]['Num_Ones']

            # If next slice is smaller than the threshold or empty
            if num_ones_ip1 <= thr:

                # And previous slice is smaller than the threshold or empty
                if num_ones_im1 <= thr:
                    #Make current slice empty
                    Keep = False
                # Keep current slice
                else:
                    Keep = True
            # If next slice is larger than the threshold
            else:
                # Keep current slice
                Keep = True
    #If current slice is larger than the threshold
    else:
        Keep=True

    return Keep, slice_info
#Function to check isolated pixels (prediction errors) on slices that contain the bladder
def contour_filtering(pred_mask_dir,df_time_phase, i, thr):
    slice_info = df_time_phase.iloc[i]
    pred_mask = Image.open(os.path.join(pred_mask_dir,slice_info['Mask_Name'])).convert('L')
    pred_arr = (np.array(pred_mask)/255).astype(np.uint8)
    #Creates a mask that holds information on non-connected countours in one 2D plane
    labeled_mask, _ = ndimage.label(pred_arr)
    #Holds information on the amount of pixels of each contour 
    component_sizes = slice_info['Component_Sizes']
    #Set the number of pixels in the background to 0 so the background is never chosen
    component_sizes[0] = 0
    #Check the number of pixels in each contour and provides a list with the contours with more pixels than the threshold
    largest_labels = (np.argwhere(component_sizes>=thr)).ravel()
    #Create a blank array
    filtered_arr = np.zeros_like(pred_arr)
    #For each contour larger thant he threshold, add that contour to a blank array
    for label in largest_labels:
        filtered_arr += (labeled_mask == label).astype(np.uint)
    #Multiply the array by 255 and return a binary image from the array
    filtered_arr = filtered_arr*255
    filtered_image = Image.fromarray(filtered_arr)
    return filtered_image

def get_args():
    parser = argparse.ArgumentParser(description='Filtered isolated pixels from predicted masks')
    parser.add_argument('--pred-dir', '-i', metavar='PRED_DIR', help='Predicted mask directory', required=True)
    parser.add_argument('--filtered-dir', '-o', metavar='FILTERED_DIR', help='Filtered predicted mask directory', required=True)
    parser.add_argument('--thr', '-t', metavar='THR',  type=int, default=80, dest='thr',
                        help='Threshold for number of pixels to filter')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    pred_masks_dir = args.pred_dir
    filtered_masks_dir = args.filtered_dir
    thr = args.thr

    Mask_Name_List = []
    Time_Phase_List =[]
    Slice_List = []
    Num_Ones_List = []
    Contours_List = []
    
    # Get time phase, slice number, number of ones and number of contours in predicted masks
    for mask_name in sorted(os.listdir(pred_masks_dir)):
        pred_mask_name = os.path.join(pred_masks_dir, mask_name)
        pred_mask = Image.open(pred_mask_name).convert('L')
        pred_arr = (np.array(pred_mask)/255).astype(np.uint8)
        labeled_mask, _ = ndimage.label(pred_arr)
        component_sizes = np.bincount(labeled_mask.ravel())
        unique_values,num_pixels = np.unique(pred_arr, return_counts=True)
        Mask_Name_List.append(mask_name)
        Time_Phase_List.append(int(mask_name.split("t")[1].split('_')[0]))
        Slice_List.append(int(mask_name.split("z")[1].split('.')[0]))
        Num_Ones_List.append(256**2 - num_pixels[0])
        Contours_List.append(component_sizes) 
    
    #Create data frame
    data = {'Mask_Name': Mask_Name_List,
            'Time_Phase': Time_Phase_List,
            'Slice': Slice_List,
            'Num_Ones': Num_Ones_List,
            'Component_Sizes':Contours_List}
    df = pd.DataFrame(data)

    # For every time phase
    for time_phase in range(1,max(df['Time_Phase'])+1):

        # Data frame for each time phase
        df_time_phase = df[df['Time_Phase']==time_phase]

        N_slices = len(df_time_phase)
        edge_indices = set(range(5)) | set(range(N_slices-5, N_slices))
        
        for i in range(N_slices):

            # Filter isolated predictions for every slice
            Keep, slice_info = check_isolation(df_time_phase,i,thr)

            # Empty mask for edge slices
            if i in edge_indices:
                new_img = Image.fromarray(np.zeros((256,256)))
                new_img.save(os.path.join(filtered_masks_dir, slice_info['Mask_Name']))

            # Middle slices
            else:
                if Keep:
                    # Check if the image has more than one contour (>2 because the background is a contour) 
                    if len(df_time_phase.iloc[i]['Component_Sizes']>2):
                        #Perform contour filtering
                        filtered_image =  contour_filtering(pred_masks_dir,df_time_phase,i,thr)
                        #Save filtered image with contours larger than the threshold
                        filtered_image.save(os.path.join(filtered_masks_dir,slice_info['Mask_Name']))
                    # Copy predicted mask to filtered mask directory
                    else:
                        shutil.copy(os.path.join(pred_masks_dir, slice_info['Mask_Name']),
                                os.path.join(filtered_masks_dir, slice_info['Mask_Name']))   
                # Create empty mask instead of predicted mask for images that are isolated pixels (prediction errors)
                else:
                    new_img = Image.fromarray(np.zeros((256,256)))
                    new_img.save(os.path.join(filtered_masks_dir, slice_info['Mask_Name']))
