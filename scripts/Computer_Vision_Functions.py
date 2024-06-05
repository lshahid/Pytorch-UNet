import os, pdb
import numpy as np
import scipy as sc
from skimage import measure
from PIL import Image
from stl import mesh
import imageio.v3 as iio

epsilon = 1e-10

def Horizontal_Flip(img_path,img_save_folder):
    img = Image.open(img_path)
    flipped_image = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img_name = img_path.split('/')[2].split('.')[0]
    img_file_extension = img_path.split('/')[2].split('.')[1]
    img_save_path = img_save_folder + img_name + '_Flipped.' + img_file_extension
    return flipped_image.save(img_save_path)

def Image_Rotation(img_path,rotation_degree,img_save_folder):
    img = Image.open(img_path)
    rotated_image = img.rotate(rotation_degree)
    img_name = img_path.split('/')[2].split('.')[0]
    img_file_extension = img_path.split('/')[2].split('.')[1]
    img_save_path = img_save_folder + img_name + '_Rotated.' + img_file_extension
    return rotated_image.save(img_save_path)

def Mask_Rotation(mask_path,rotation_degree,mask_save_folder):
    mask = Image.open(mask_path)
    rotated_mask = mask.rotate(rotation_degree,fillcolor=1)
    mask_name = mask_path.split('/')[2].split('.')[0]
    mask_file_extension = mask_path.split('/')[2].split('.')[1]
    mask_save_path = mask_save_folder + mask_name + '_Rotated.' + mask_file_extension
    return rotated_mask.save(mask_save_path)

def Hausdorff_Distance(original_mask_path,predicted_mask_path):
    Original = Image.open(original_mask_path)
    Predicted = Image.open(predicted_mask_path)
    Original = np.array(Original)
    Predicted = np.array(Predicted)
    Distance,orig_idx,predicted_idx = sc.spatial.distance.directed_hausdorff(Original,Predicted)
    return Distance, orig_idx, predicted_idx

def DICE_Coeff(original_mask_path,predicted_mask_path):
    Original = Image.open(original_mask_path)
    Predicted = Image.open(predicted_mask_path)

    Original = np.array(Original)
    Predicted = np.array(Predicted)
    if np.count_nonzero(Original)!=0:
        Original = np.where((Original==0)|(Original==1),Original^1,Original)
    Predicted = Predicted / 255
    Intersection = np.sum(Original * Predicted)
    DICE = (2. * Intersection + epsilon)  / (np.sum(Original) + np.sum(Predicted) + epsilon)
    return DICE

def Mask_to_Array(file_path):
    img = Image.open(file_path)
    array = np.array(img)

    if array[0,0] != 0:
        indices_zero = array == 0
        indices_one = array == 1

        array[indices_one] = 0
        array[indices_zero] = 1

    return array

def Transform_Image_to_2D_FFT(file_path,image_save_path,desired_file_ext):
    if file_path.split('.')[-1] == 'gif':
        img = iio.imread(file_path)
        Img_Avg = img[0].mean(axis=2)
        Img_FFT = sc.fft.fft2(Img_Avg)
        Img_FFT = sc.fft.fftshift(Img_FFT)
        Img_FFT_Magnitude = np.log(abs(Img_FFT))
        Normalized_Img_FFT_Magnitude = (255*Img_FFT_Magnitude/np.max(Img_FFT_Magnitude)).astype(np.uint8)
        File_Name = (file_path.split('.')[-2]).split('/')[-1]
        Save_Path = image_save_path + File_Name + '.' + desired_file_ext
        return iio.imwrite(Save_Path,Normalized_Img_FFT_Magnitude)
    else:
        img = iio.imread(file_path)
        Img_Avg = img.mean(axis=2)
        Img_FFT = sc.fft.fft2(Img_Avg)
        Img_FFT = sc.fft.fftshift(Img_FFT)
        Img_FFT_Magnitude = np.log(abs(Img_FFT))
        Normalized_Img_FFT_Magnitude = (255*Img_FFT_Magnitude/np.max(Img_FFT_Magnitude)).astype(np.uint8)
        File_Name = (file_path.split('.')[-2]).split('/')[-1]
        Save_Path = image_save_path + File_Name + '.' + desired_file_ext
        return iio.imwrite(Save_Path,Normalized_Img_FFT_Magnitude)


def Transform_2D_FFT_to_Image(Image_FFT,image_save_path):
    if str(type(Image_FFT)[0])=='numpy.ndarray[0]':
        ifft = sc.fft.ifftshift(Image_FFT)
        ifft = sc.fft.ifft2(ifft)
        Reconstructed_Img = np.abs(ifft)
        return iio.imwrite(image_save_path,Reconstructed_Img)
    else:
        if Image_FFT.split('.')[-1] == 'gif':
            img = iio.imread(Image_FFT)

def Array_to_STL(array):
    
    verts,faces,normals,values = measure.marching_cubes(array)
    stl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        stl.vectors[i] = verts[f]
    return stl
