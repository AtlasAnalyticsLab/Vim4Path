import os
import h5py
import glob
import openslide
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import argparse


def process_patch(patch_path):
    class_name = patch_path.split(os.sep)[-3]  # Adjust index based on the structure of patch_path
    file_name = os.path.basename(patch_path)
    patch_name = os.path.splitext(file_name)[0]
    
    raw_patch_path = os.path.join(args.raw_data_folder, class_name, f"{patch_name}.{args.wsi_extension}")
    wsi = openslide.open_slide(raw_patch_path)
    with h5py.File(patch_path, 'r') as hdf5_file:
        patch_level = hdf5_file['coords'].attrs['patch_level']
        patch_size = hdf5_file['coords'].attrs['patch_size']
        for idx in range(len(hdf5_file['coords'])):
            coord = hdf5_file['coords'][idx]
            if os.path.isfile(os.path.join(args.output_folder, class_name, f"{patch_name}_{idx}.jpg")):
                continue
            img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
            img = np.array(img)
            cv2.imwrite(os.path.join(args.output_folder, class_name, f"{patch_name}_{idx}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main(args):
    class_names = os.listdir(args.input_folder)
    for class_name in class_names:
        os.makedirs(os.path.join(args.output_folder, class_name), exist_ok=True)
        print(f"Processing for Class {class_name}")
        if args.sample_count>0:
            patch_paths = sorted(glob.glob(os.path.join(args.input_folder, class_name, "patches", "*h5")))[
                          :args.sample_count]
        else:
            patch_paths = sorted(glob.glob(os.path.join(args.input_folder, class_name, "patches", "*h5")))
        total = len(patch_paths)
        progress_bar = tqdm(total=total)
        with Pool(min(cpu_count(), 8)) as p:
            for _ in p.imap_unordered(process_patch, patch_paths):
                progress_bar.update()
        progress_bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_folder', type=str, help='Path to the folder containing raw WSIs.')
    parser.add_argument('--wsi_extension', type=str, choices=['ndpi', 'tif', 'svs'], help='Extension of WSI file type. Valid choices are [ndpi, tif, svs]')
    parser.add_argument('--input_folder', type=str, help='Folder that contains h5 files extracted from WSI using '
                                                       'create_patches_fp.py')
    parser.add_argument('--output_folder', type=str, help='Folder to save extracted patches.')
    parser.add_argument('--sample_count', type=int, default=-1, help='Maximum number of WSIs to extract patches. If -1, it will extract all the patches.')
    args = parser.parse_args()
    main(args)