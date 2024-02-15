import os
import h5py
import glob
import openslide
from tqdm import tqdm
import webdataset as wds
from PIL import Image
import io
import json
raw_data_paths = {
    'digestive_benign': "/home/atlas-gp/Transfer_CC/10930_chx_digestive_benigne/",
    'digestive_malign': "/home/atlas-gp/Transfer_CC/10931_chx_digestive_maligne"
}

input_folder = 'output'
class_names = raw_data_paths.keys()
hipt_patch_folder = 'extracted_mag10x_patch256_fp'
output_pattern = 'output_tar/dataset-%02d.tar'  # Pattern for sharded output files

# Configuration for sharding
max_shard_size = 3e10  # Maximum shard size in bytes (e.g., 30GB)

# Create a ShardWriter object to write the dataset to sharded tar archives

total_count =0
# , maxcount=3e5
with wds.ShardWriter(output_pattern, maxsize=max_shard_size) as sink:
    for class_name in class_names:
        print(f"Processing for Class {class_name}")
        patch_h5_paths = glob.glob(os.path.join(input_folder, class_name, hipt_patch_folder, "patches", "*h5"))[:20]
        for patch_h5_path in tqdm(patch_h5_paths):
            file_name = os.path.basename(patch_h5_path)
            wsi_id = os.path.splitext(file_name)[0]
            raw_wsi_path = os.path.join(raw_data_paths[class_name], f"{wsi_id}.ndpi")
            wsi = openslide.open_slide(raw_wsi_path)

            with h5py.File(patch_h5_path, 'r') as hdf5_file:
                patch_level = hdf5_file['coords'].attrs['patch_level']
                patch_size = hdf5_file['coords'].attrs['patch_size']
                for idx, coord in enumerate(hdf5_file['coords']):
                    img = wsi.read_region((coord[0], coord[1]), patch_level, (patch_size, patch_size)).convert('RGB')                    
                    # Convert PIL image to bytes
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()
                    img_byte_arr.close()

                    # Key, class label, and coordinates for each image
                    key = f"{wsi_id}_{idx}"
                    class_label = class_name
                    coord_str = f"{coord[0]}_{coord[1]}"
                    
                    # Write the image, metadata, and coordinates to the ShardWriter
                    sink.write({
                        "__key__": key,
                        "class": class_label,
                        "coords": coord_str,
                        "jpg": img_bytes
                    })
                    total_count  += 1

print(total_count )
