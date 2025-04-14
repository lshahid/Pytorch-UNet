"""
This script segements all images in a directory.
"""
import argparse
import logging
import os
import pdb

import numpy as np
import matplotlib.image

from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description='SAU Implementation of Predicted Images')
    #parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
    #                    help='Specify the file in which the model is stored')
    parser.add_argument('--input-dir', '-i', metavar='INPUT_DIR', help='Input image directory', required=True)
    parser.add_argument('--output-dir', '-o', metavar='OUTPUT_DIR', help='Output image directory', required=True)
    parser.add_argument('--n_timephases', '-t', type=int, help='Number of time phases', required=True)
    parser.add_argument('--n_slices', '-s',  type=int, help='Number of slices', required=True)
    #parser.add_argument('--viz', '-v', action='store_true',
    #                   help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    #parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
    #                    help='Minimum probability value to consider a mask pixel white')
    #parser.add_argument('--scale', '-s', type=float, default=0.5,
    #                    help='Scale factor for the input images')
    #parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    #parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}.gif'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # in_files = args.input
    # out_files = get_output_filenames(args)
    in_files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if os.path.isfile(os.path.join(input_dir, f))]
    out_files = [os.path.join(output_dir, os.path.splitext(os.path.basename(f))[0] + '.gif') for f in in_files]

    # SAU-Net
    n_timephases = args.n_timephases
    n_slices = args.n_slices

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img_i = Image.open(filename)
        mask_i = np.array(img_i)/255
        mask_i_s = 1/(1 + np.exp(-mask_i))


        slice = i % n_slices
        if slice == 0:
            img_ip1 = Image.open(in_files[i+1])
            mask_ip1 = np.array(img_ip1)/255

            # Attention
            mask_ip1_s = 1/(1 + np.exp(-mask_ip1))

            # Multiply
            mask_i_ip1 = np.multiply(mask_i_s,  mask_ip1_s)

            # Add
            mask_add = mask_i_s + mask_i_ip1

        elif slice == n_slices-1:
            img_im1 = Image.open(in_files[i-1])
            mask_im1 = np.array(img_im1)/255

            # Attention
            mask_im1_s = 1/(1 + np.exp(-mask_im1))

            # Multiply
            mask_i_im1 = np.multiply(mask_i_s,  mask_im1_s)

            # Add
            mask_add = mask_i_s + mask_i_im1
            

        else:
            img_im1 = Image.open(in_files[i-1])
            mask_im1 = np.array(img_im1)/255
            img_ip1 = Image.open(in_files[i+1])
            mask_ip1 = np.array(img_ip1)/255

            # Attention
            mask_im1_s = 1/(1 + np.exp(-mask_im1))
            mask_ip1_s = 1/(1 + np.exp(-mask_ip1))

            # Multiply
            mask_i_im1 = np.multiply(mask_i_s,  mask_im1_s)
            mask_i_ip1 = np.multiply(mask_i_s,  mask_ip1_s)

            # Add
            mask_add = mask_i_s + mask_i_im1 + mask_i_ip1

        # Sigmoid
        mask_sigmoid = 1/(1 + np.exp(-mask_add))
        dynamic_thresh = np.unique(mask_sigmoid)[0]

        if not args.no_save:
            out_filename = out_files[i]
            result = np.where(mask_sigmoid > dynamic_thresh, 1, 0)
            matplotlib.image.imsave(out_filename,result,cmap='gray')
            logging.info(f'Mask saved to {out_filename}')