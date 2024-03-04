# import os
# import h5py
# import glob 
# import openslide
# from tqdm import tqdm
# raw_data_paths = {'digestive_benign': "/home/atlas-gp/Transfer_CC/10930_chx_digestive_benigne/",
#                   'digestive_malign': "/home/atlas-gp/Transfer_CC/10931_chx_digestive_maligne"}

# output_folder = 'output_hipt'

# input_folder = 'output'
# class_names = os.listdir(input_folder)

# hipt_patch_folder = 'extracted_mag10x_patch1280_fp'


# for class_name in class_names:
#     os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

#     print(f"Processing for Class {class_name}")
#     patch_h5_paths = glob.glob(os.path.join(input_folder, class_name, hipt_patch_folder, "patches", "*h5"))[:20]
#     for patch_h5_path in tqdm(patch_h5_paths):
#         file_name = os.path.basename(patch_h5_path)
#         wsi_id = os.path.splitext(file_name)[0]
#         raw_wsi_path = os.path.join(raw_data_paths[class_name], f"{wsi_id}.ndpi")
#         wsi = openslide.open_slide(raw_wsi_path)

#         with h5py.File(patch_h5_path,'r') as hdf5_file:
#             patch_level = hdf5_file['coords'].attrs['patch_level']
#             patch_size = hdf5_file['coords'].attrs['patch_size']
#             for idx in range(len(hdf5_file['coords'])):
#                 coord = hdf5_file['coords'][idx]
#                 img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
#                 img.save(os.path.join(output_folder, class_name, f"{wsi_id}_{idx}.jpg"))
import os
import h5py
import glob
import openslide
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np 
raw_data_paths = {
    'digestive_benign': "/home/atlas-gp/Transfer_CC/10930_chx_digestive_benigne/",
    'digestive_malign': "/home/atlas-gp/Transfer_CC/10931_chx_digestive_maligne"
}

output_folder = '/home/a_n29343/CHUM/VIM4Path/datasets/CHUM/output_vim1280_test/'
input_folder = '/home/a_n29343/CHUM/VIM4Path/datasets/CHUM/output'
class_names = os.listdir(input_folder)
hipt_patch_folder = 'extracted_mag5x_patch1280_fp'

def process_patch(patch_path):

    class_name = patch_path.split(os.sep)[-4]  # Adjust index based on the structure of patch_path
    file_name = os.path.basename(patch_path)
    patch_name = os.path.splitext(file_name)[0]
    
    raw_patch_path = os.path.join(raw_data_paths[class_name], f"{patch_name}.ndpi")
    wsi = openslide.open_slide(raw_patch_path)
    with h5py.File(patch_path, 'r') as hdf5_file:
        patch_level = hdf5_file['coords'].attrs['patch_level']
        patch_size = hdf5_file['coords'].attrs['patch_size']
        for idx in range(len(hdf5_file['coords'])):
            coord = hdf5_file['coords'][idx]
            if os.path.isfile(os.path.join(output_folder, class_name, f"{patch_name}_{idx}.jpg")):
                continue
            img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
            img = np.array(img)
            # if img.shape[0]>0:
            img = cv2.imwrite(os.path.join(output_folder, class_name, f"{patch_name}_{idx}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # img.save(os.path.join(output_folder, class_name, f"{patch_name}_{idx}.jpg"))


def main():
    for class_name in class_names:
        os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)
        print(f"Processing for Class {class_name}")
        patch_paths = sorted(glob.glob(os.path.join(input_folder, class_name, hipt_patch_folder, "patches", "*h5")))#[400:600]
        total = len(patch_paths)
        progress_bar = tqdm(total=total)
        with Pool(min(cpu_count(), 8)) as p:
            for _ in p.imap_unordered(process_patch, patch_paths):
                progress_bar.update()
        progress_bar.close()
if __name__ == "__main__":
    main()