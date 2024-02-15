import os
import h5py
import glob 
import openslide
from tqdm import tqdm
raw_data_paths = {'digestive_benign': "/home/atlas-gp/Transfer_CC/10930_chx_digestive_benigne/",
                  'digestive_malign': "/home/atlas-gp/Transfer_CC/10931_chx_digestive_maligne"}

input_folder = 'preprocess/output'
class_names = os.listdir(input_folder)
hipt_patch_folder = 'extracted_mag10x_patch256_fp'

for class_name in class_names:
    total_count = 0
    print(f"Processing for Class {class_name}")
    patch_paths = sorted(glob.glob(os.path.join(input_folder, class_name, hipt_patch_folder, "patches", "*h5")))#[:300]
    print(patch_paths[0])
    for patch_path in tqdm(patch_paths):
        file_name = os.path.basename(patch_path)
        
        with h5py.File(patch_path,'r') as hdf5_file:
            patch_level = hdf5_file['coords'].attrs['patch_level']
            patch_size = hdf5_file['coords'].attrs['patch_size']
            patch_count = len(hdf5_file['coords'])
            total_count += patch_count

    print (f"Total patch count for {class_name} is {total_count}")