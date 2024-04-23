# VIM4Path
PyTorch Implementation of VIM4Path paper and pretrained weights.  

## Installation

Use the installation guide on  [Vision Mamba Repo](https://github.com/hustvl/vim).
Also, need to install packages such as shapely, openslide, opencv, h5py, and lxml for data processing. 


## Dataset

#### Dataset Source
Camelyon16 WSI images can be downloaded from the following FTP site:
[CAMELYON16 Dataset FTP](https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/)

### Data processing 
You should use the preprocess folder where we integrated [CLAM](https://github.com/mahmoodlab/CLAM) preprocessing code. 


To create patches at 10x zooming level, we can use the following commands:
```python
python create_patches_fp.py --source path_to_Camelyon16/testing/images/ --save_dir ../dataset/Camelyon16/testing/224_10x/h5/ --patch_size 224 --step_size 224 --patch_level 2 --seg --patch --stitch
python create_patches_fp.py --source path_to_Camelyon16/training/normal/ --save_dir ../dataset/Camelyon16/training/224_10x/h5/normal/ --patch_size 224 --step_size 224 --patch_level 2 --seg --patch --stitch
python create_patches_fp.py --source path_to_Camelyon16/training/tumor/ --save_dir ../dataset/Camelyon16/training/224_10x/h5/tumor/ --patch_size 224 --step_size 224 --patch_level 2 --seg --patch --stitch
```

Use the extract_patches.py script for pretraining image extraction with the following command:
```python 
python extract_patches.py --raw_data_folder path_to_raw_WSIs --wsi_extension tif --input_folder path_to_h5_files --output_folder path_to_save_patches
```

To extract patches for patch-level classification use the camelyon16_extraction.ipynb.

## Pretraining
You should use the dino folder for pretraining. 

For pretraining code you can use the following command. Make sure to have a total batch size of 512 across all GPUs similar to the paper. You can ignore "disable_wand" if you want to use W&B to track your experiments. 
```python
python -m torch.distributed.launch --nproc_per_node=4 main.py --data_path patch_to_pretraining_images --output_dir checkpoints/camelyon16_224_10x/vim-s/ --image_size 224 --image_size_down 96 --batch_size_per_gpu 128 --arch vim-s --disable_wand
```


## Patch-Level Evaluation 
You can use the following command to evaluate each model's performance on extracted patch-level images (using camelyon16_extraction.ipynb). We use batch_size of 64 and train for 20 epochs since all methods tend to overfit after this number of epochs. 
```python
python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py --output_dir checkpoints/camelyon16_224_10x/vim-s/eval_linear --train_data_path path_to_balanced_pcam10x_data --val_data_path /data2/projects/VIM4Path/datasets/Camelyon16/Cam16_Balanced/Balanced/224_5x/test/ --pretrained_weights checkpoints/camelyon16_224_10x/vim-s/checkpoint.pth --arc vim-s  --image_size 224 --epochs 20  --batch_size 64 --
disable_wand
```

## Slide-Level Evaluation 
For slide-level classification you can use the following command to get the features for slide at 10x using the pretrained model at 10x. 
```python
python mil_data_creation.py --image_size 224 --arch vim-s --pretrained_weights dino/checkpoints/camelyon16_224_10x/vim-s_224-96/checkpoint.pth --source_level 10 --target_level 10
```

We modify the [CLAM](https://github.com/mahmoodlab/CLAM) code to work on our dataset. So, you can use the following command in the MIL folder to get slide-level performance.
```python
python main_cam.py  --image_size 224 --arch vim-s --source_level 10 --target_level 10 --exp_code vim-s-224-10at10-clam_sb --model_type clam_sb --drop_out --early_stopping --lr 2e-4 --k 1 --label_frac 1  --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --log_data
```


# Weights
The pretrained weights and the self-supervised logs are provided below.
<table>
  <tr>
    <td>arch</td>
    <th colspan="2">download</th>
  </tr>
  <tr>
    <td>ViT-ti</td>
    <td><a href="https://www.dropbox.com/scl/fo/9rmze3a0u0rmfvv4uogby/AN25BiCsNh0o3rnnA9dYmNQ?dl=0&e=1&preview=checkpoint.pth&rlkey=ufc80pc2spzc98cn4atrh26jl">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/9rmze3a0u0rmfvv4uogby/AN25BiCsNh0o3rnnA9dYmNQ?dl=0&e=1&preview=log.txt&rlkey=ufc80pc2spzc98cn4atrh26jl">pretraining log</a></td>
  </tr>
  <tr>
    <td>ViT-s</td>
    <td><a href="https://www.dropbox.com/scl/fo/z1w40ypwbsyqlkywevm5t/AFAnxqw0VPnRmf8c1KoulKU?dl=0&e=1&preview=checkpoint.pth&rlkey=vq3xq6dj4hmtrv1qeah1cnmlg">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/z1w40ypwbsyqlkywevm5t/AFAnxqw0VPnRmf8c1KoulKU?dl=0&e=1&preview=log.txt&rlkey=vq3xq6dj4hmtrv1qeah1cnmlg">pretraining log</a></td>
  </tr>

  <tr>
    <td>Vim-ti</td>
    <td><a href="https://www.dropbox.com/scl/fo/4q86hsyhxqf0s30sznsi7/AD3K7kL0D9tMCEw2s6GzOGs?dl=0&e=1&preview=checkpoint.pth&rlkey=57wabu98dei6x60u6dxhe33vg">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/4q86hsyhxqf0s30sznsi7/AD3K7kL0D9tMCEw2s6GzOGs?dl=0&e=1&preview=log.txt&rlkey=57wabu98dei6x60u6dxhe33vg">pretraining log</a></td>
  </tr>

  <tr>
    <td>Vim-ti-plus</td>
    <td><a href="https://www.dropbox.com/scl/fo/93486j0plk4zz185ncmio/AEeLYVl1Cv92ucHYnc5zAXc?dl=0&e=1&preview=checkpoint.pth&rlkey=2scjj7ekkceii4iepexg0huvx">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/93486j0plk4zz185ncmio/AEeLYVl1Cv92ucHYnc5zAXc?dl=0&e=1&preview=log.txt&rlkey=2scjj7ekkceii4iepexg0huvx">pretraining log</a></td>
  </tr>

  <tr>
    <td>Vim-s</td>
    <td><a href="https://www.dropbox.com/scl/fo/itlxf4cqyvxrbp7kxh43t/ADEuaFPA4Fv5Le96B2T1YZk?dl=0&e=1&preview=checkpoint.pth&rlkey=tf9du6jleuvymfcbhsi67iuf2">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/itlxf4cqyvxrbp7kxh43t/ADEuaFPA4Fv5Le96B2T1YZk?dl=0&e=1&preview=log.txt&rlkey=tf9du6jleuvymfcbhsi67iuf2">pretraining log</a></td>
  </tr>

</table>

## Citation
If you find this repository useful, please consider giving a star and citation:
```