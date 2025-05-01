#Turn Masks output from the Automatic Segmentation Model into STLs or Part Files to inport into Mimics
from glob import glob
import os, pdb, argparse, shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from stl import mesh
from skimage import measure


#Functions
def Mask_to_Array(file_path):
    img = Image.open(file_path)
    array = np.array(img)

    if array[0,0] != 0:
        indices_zero = array == 0
        indices_one = array == 1

        array[indices_one] = 0
        array[indices_zero] = 1

    return array

def Array_to_STL(array):
   
    verts,faces,normals,values = measure.marching_cubes(array)
    stl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        stl.vectors[i] = verts[f]
    return stl


def get_args():
    parser = argparse.ArgumentParser(description='Filtered isolated pixels from predicted masks')
    parser.add_argument('--mask-dir', '-i', metavar='MASK_DIR', help='Predicted mask directory', required=True)
    parser.add_argument('--stl-dir', '-o', metavar='STL_DIR', help='STL directory', required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    mask_dir = args.mask_dir
    stl_dir = args.stl_dir

    mask_list = sorted(os.listdir(mask_dir))
    
    Mask_Name_List = []
    Time_Phase_List =[]
    Slice_List = []

    for mask_name in sorted(os.listdir(mask_dir)):
        pred_mask_name = os.path.join(mask_dir, mask_name)
        Mask_Name_List.append(mask_name)
        Time_Phase_List.append(int(mask_name.split("t")[1].split('_')[0]))
        Slice_List.append(int(mask_name.split("z")[1].split('.')[0]))

    data = {'Mask_Name': Mask_Name_List,
            'Time_Phase': Time_Phase_List,
            'Slice': Slice_List}
    df = pd.DataFrame(data)

    for time_phase in pd.unique(df['Time_Phase']):
        
        df_Time_Phase = df[df['Time_Phase']==time_phase]
        Subject_Name = df_Time_Phase.iloc[0]['Mask_Name'].split('_')[0]
        Phase_Number = "t{:03d}".format(time_phase)

        Array_3D = np.empty(shape=(256,256,len(df_Time_Phase)))

        for i in range(len(df_Time_Phase)):
            Slice_Info = df_Time_Phase.iloc[i]
            Mask_Name = Slice_Info['Mask_Name']
            Array_2D = Mask_to_Array(os.path.join(mask_dir,Mask_Name))
            Array_3D[:,:,i] = Array_2D 
        stl = Array_to_STL(Array_3D)
        
        if os.path.exists(stl_dir) is False:
            os.makedirs(stl_dir)
        output_stl_name = '{:s}_{:s}.stl'.format(os.path.join(stl_dir,Subject_Name),Phase_Number)
        stl.save(output_stl_name)


    # STL_Output_Folder = STL_Output_Folder + Subject_Name
    # STL_Output_Exists = os.path.exists(STL_Output_Folder)

    # if STL_Output_Exists is False:
    #     os.makedirs(STL_Output_Folder)

    # stl.save('{:s}/{:s}_{:s}.stl'.format(STL_Output_Folder,Subject_Name,Phase_Number))