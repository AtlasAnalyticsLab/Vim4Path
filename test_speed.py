import os
import time
import psutil
import h5py
import glob
import openslide
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import webdataset as wds
from PIL import Image
import io

# Function to print current memory usage
def print_memory_usage(description):
    process = psutil.Process(os.getpid())
    print(f"{description}: {process.memory_info().rss / 1024 ** 2:.2f} MB")
def test_image_loader():
    # Define your transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Initializing Dataset ImageFolder")
    # Load your dataset with the transform
    dataset = datasets.ImageFolder('preprocess/output_hipt_jpg', transform=transform)
    print(f"Loaded dataset with length :{len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)



    # Print initial memory usage
    print_memory_usage("Memory usage before loading dataset")

    # Start timing
    start_time = time.time()

    # Simulate one epoch of data loading
    for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
        # Optionally, print memory usage at each step or after certain intervals
        if batch_idx % 100 == 0:  # Adjust the interval as needed
            print_memory_usage(f"Memory usage at batch {batch_idx}")

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Print final memory usage
    print_memory_usage("Memory usage after loading dataset")

    print(f"Total time to load all samples for one epoch: {total_time} seconds")

def test_wsi_loader():

    # Define paths to the WSIs
    raw_data_paths = {
        'digestive_benign': "/home/atlas-gp/Transfer_CC/10930_chx_digestive_benigne/",
        'digestive_malign': "/home/atlas-gp/Transfer_CC/10931_chx_digestive_maligne/"
    }

    # Define the input folder and high-importance patch folder (assuming these are correct based on your setup)
    input_folder = 'preprocess/output'
    hipt_patch_folder = 'extracted_mag10x_patch256_fp'

    # Define your transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Instantiate the dataset class
    dataset = Whole_Slide_Bag_FP_MultiWSI(
        raw_data_paths=raw_data_paths,
        input_folder=input_folder,
        hipt_patch_folder=hipt_patch_folder,
        pretrained=False,  # Assuming you're not using pretrained transforms
        transforms=transform,  # You can define custom transformations here
        custom_downsample=1,  # Assuming no downsample
        target_patch_size=-1  # Assuming you're not resizing patches
    )

    print(f"Loaded dataset with {len(dataset)} patches")



    # Load your dataset with the transform
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Print initial memory usage
    print_memory_usage("Memory usage before loading dataset")

    # Start timing
    start_time = time.time()

    # Simulate one epoch of data loading
    for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
        # Optionally, print memory usage at each step or after certain intervals
        if batch_idx % 100 == 0:  # Adjust the interval as needed
            print_memory_usage(f"Memory usage at batch {batch_idx}")

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Print final memory usage
    print_memory_usage("Memory usage after loading dataset")

    print(f"Total time to load all samples for one epoch: {total_time} seconds")

class Whole_Slide_Bag_FP_MultiWSI(Dataset):
    def __init__(self, raw_data_paths, input_folder, hipt_patch_folder, pretrained=False, transforms=None, custom_downsample=1, target_patch_size=-1):
        """
        Args:
            raw_data_paths (dict): Dictionary with class names as keys and paths to folders containing WSIs as values.
            input_folder (str): Base input folder path for processed data.
            hipt_patch_folder (str): Subfolder within each class directory containing patch files.
            pretrained (bool): Use ImageNet transforms.
            transforms (callable, optional): Optional transform to be applied on a sample.
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size).
            target_patch_size (int): Custom defined image size before embedding.
        """
        self.pretrained = pretrained
        self.transforms = transforms
        self.custom_downsample = custom_downsample
        self.patches_info = {'wsi_paths':[], 'patch_coords':[], 'classes':[]}

        # Load WSI and corresponding patches
        for class_name in raw_data_paths.keys():
            path_h5_folder = os.path.join(input_folder, class_name, hipt_patch_folder, "patches")
            patch_h5_paths = list(sorted(glob.glob(os.path.join(path_h5_folder, "*.h5"))))[:20]
            for patch_h5_path in tqdm(patch_h5_paths):
                with h5py.File(patch_h5_path,'r') as hdf5_file:
                    coords = list(hdf5_file['coords'])
                    self.patches_info['patch_coords'].extend(coords)
                
                self.patches_info['classes'].extend([class_name]*len(coords))

                wsi_id = os.path.splitext(os.path.basename(patch_h5_path))[0]
                wsi_path = os.path.join(raw_data_paths[class_name], f"{wsi_id}.ndpi")
                self.patches_info['wsi_paths'].extend([wsi_path]*len(coords))

        
        # get patch level and patch size
        with h5py.File(patch_h5_path,'r') as hdf5_file:
            self.patch_level = hdf5_file['coords'].attrs['patch_level']
            self.patch_size = hdf5_file['coords'].attrs['patch_size']

    def __len__(self):
        return len(self.patches_info['wsi_paths'])

    def __getitem__(self, idx):

        wsi = openslide.open_slide(self.patches_info['wsi_paths'][idx])
        coord = self.patches_info['patch_coords'][idx]
        class_label = self.patches_info['classes'][idx]
        img = wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.transforms(img)
        return img, class_label


# Assuming these are your class labels
class_labels = ['digestive_benign', 'digestive_malign']
# Create a mapping from class labels to integers
label_to_int = {label: idx for idx, label in enumerate(class_labels)}
import torch
import cv2
import numpy as np
# # Custom processing function to handle each sample
# def process_sample(sample):
#     # Define your transform
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     # Decode the image
#     image = sample['jpg']
#     image = Image.open(io.BytesIO(image)).convert("RGB")
#     image = transform(image)
    
#     # Decode the class label
#     cls = label_to_int[sample['class'].decode('utf-8')]
    
#     # Return the processed sample
#     return {'jpg': image, 'cls': cls}

# Optimized custom processing function using OpenCV for image loading
def process_sample(sample):
    # OpenCV for faster image decoding
    image = cv2.imdecode(np.frombuffer(sample['jpg'], np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Define your transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transformation
    image = transform(image)

    # Decode the class label
    cls = label_to_int[sample['class'].decode('utf-8')]
    
    return {'jpg': image, 'cls': cls}


def test_webdataset_loader():
    tar_path = 'preprocess/output_tar/dataset-00.tar'
    # Define your transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Adjust the dataset creation to match the stored keys: "jpg" for images, "cls" for labels
    # Since "coords" is also stored, you might want to use it in some way. Here, we're not using it directly in the model,
    # but you could modify this part to include it in your processing pipeline.
    dataset = wds.WebDataset(tar_path).shuffle(1000).map(process_sample)
    # Create a DataLoader without shuffle option
    data_loader = DataLoader(dataset, batch_size=32, num_workers=4)

    # Print initial memory usage
    print_memory_usage("Memory usage before loading dataset")

    # Start timing
    start_time = time.time()
    total_samples = 0
    # Simulate one epoch of data loading
    for batch_idx, data in enumerate(tqdm(data_loader)):
        inputs, labels = data['jpg'], data['cls']
        total_samples += inputs.shape[0]
        # Optionally, print memory usage at each step or after certain intervals
        if batch_idx % 100 == 0:  # Adjust the interval as needed
            print_memory_usage(f"Memory usage at batch {batch_idx}")
    print(total_samples)
    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Print final memory usage
    print_memory_usage("Memory usage after loading dataset")

    print(f"Total time to load all samples for one epoch: {total_time} seconds")

test_webdataset_loader()
# test_wsi_loader()
# test_image_loader()