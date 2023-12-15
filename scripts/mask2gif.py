"""
This script convert Mimics mask .jpg files to binary mask .gif files.
Note that directories and file names should be adjusted for each subject.
"""

import os
from PIL import Image
import numpy
import matplotlib.pyplot as plt
import imageio

# Image directory
img_dir = '../bladder_data/imgs_wmasks/LUTS005/'
gif_dir = '../bladder_data/masks/LUTS005/'

# Iterate over files in image directory
for img_name in os.listdir(img_dir):
    jpg_name = os.path.join(img_dir, img_name)

    # Import image as grayscale array
    img = Image.open(jpg_name).convert('L')
    gray_arr = numpy.array(img)
    
    # # Plot gray
    # plt.imshow(gray_arr, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()
    
    # Convert to binary array
    thresh_min = 50
    thresh_max = 250
    bin_arr = (gray_arr >= thresh_min) & (gray_arr <= thresh_max)
    bin_arr[200:, :] = 0
    bin_arr = bin_arr.astype('uint8')*255
    
    # # Plot binary
    # plt.imshow(bin_arr, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()
    
    # Save as gif
    gif_name = gif_dir + img_name[:4] + '0' + img_name[5:17] \
        + img_name[20:24] + '.gif'
    imageio.imsave(gif_name, bin_arr)
