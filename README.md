# VIM4Path
Vim4Path: Self-Supervised Vision Mamba for Histopathology Images, CVPR 2024.

**Abstract,**  *Representation learning from Gigapixel Whole Slide Images (WSI) poses a significant challenge in computational pathology due to the complicated nature of tissue structures and the scarcity of labeled data. Multi-instance learning methods have addressed this challenge, leveraging image patches to classify slides utilizing pretrained models using Self-Supervised Learning (SSL) approaches. The performance of both SSL and MIL methods relies on the architecture of the feature encoder. This paper proposes leveraging the Vision Mamba (Vim) architecture, inspired by state space models, within the DINO framework for representation learning in computational pathology. We evaluate the performance of Vim against Vision Transformers (ViT) on the Camelyon16 dataset for both patch-level and slide-level classification. Our findings highlight Vim’s enhanced performance compared to ViT, particularly at smaller scales, where Vim achieves an 8.21 increase in ROC AUC for models of similar size. An explainability analysis further highlights Vim’s capabilities, which reveals that Vim uniquely emulates the pathologist workflow—unlike ViT. This alignment with human expert analysis highlights Vim’s potential in practical diagnostic settings and contributes significantly to developing effective representation-learning algorithms in computational pathology.*

[[`arXiv`](https://arxiv.org/pdf/2404.13222.pdf)] | [[`Cite`]](#citation) 


![Vim4Path](media/Vim4Path.webp)




## Installation
We use cuda 11.8 for our codes. Use the following list of commands to install required libraries. 

```commandline
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
cd ../mamba-1p1p1
pip install -e .
pip install causal_conv1d==1.1.0
pip install shapely
pip install openslide-python
pip install opencv-python
pip install h5py
pip install lxml
pip install timm
```

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

Note, that there are some issues (still unfixed) with extract_patches.py where a few of the extracted images are truncated. We use the following script to remove those images. 
```python
cd preprocess
python check_images.py --dir path_to_saved_patches
```

These scripts would create patches for pretraining. Although the images are divided into two folders based on the class, those folders cannot be used for patch-level classification (since the images for the tumor class are extracted from all regions of the slide). 

To extract patches for patch-level classification use the camelyon16_extraction.ipynb, which allows extracting images for tumor class from the ROI region. a

## Pretraining
You should use the dino folder for pretraining. 

For pretraining code you can use the following command. Make sure to have a total batch size of 512 across all GPUs similar to the paper. You can ignore "disable_wand" if you want to use W&B to track your experiments. 
```python
python -m torch.distributed.launch --nproc_per_node=4 main.py --data_path patch_to_pretraining_images --output_dir checkpoints/camelyon16_224_10x/vim-s/ --image_size 224 --image_size_down 96 --batch_size_per_gpu 128 --arch vim-s --disable_wand
```


## Patch-Level Evaluation 
You can use the following command to evaluate each model's performance on extracted patch-level images (using camelyon16_extraction.ipynb). We use batch_size of 64 and train for 20 epochs since all methods tend to overfit after this number of epochs. For generating the balanced dataset, use ''camelyon16_extraction.ipynb''. 
```python
python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py --output_dir checkpoints/camelyon16_224_10x/vim-s/eval_linear --train_data_path path_to_balanced_pcam10x_train_data --val_data_path path_to_balanced_pcam10x_test_data --pretrained_weights checkpoints/camelyon16_224_10x/vim-s/checkpoint.pth --arc vim-s  --image_size 224 --epochs 20  --batch_size 64 --
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
The pretrained (no labels) weights and the self-supervised logs are provided below.
<table>
  <tr>
    <td>arch</td>
    <th>ROC AUC <br> (Cam16)</th>
    <th colspan="2">download</th>
  </tr>
  <tr>
    <td>ViT-ti</td>
    <th> 87.60 </th>
    <td><a href="https://www.dropbox.com/scl/fo/9rmze3a0u0rmfvv4uogby/AN25BiCsNh0o3rnnA9dYmNQ?dl=0&e=1&preview=checkpoint.pth&rlkey=ufc80pc2spzc98cn4atrh26jl">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/9rmze3a0u0rmfvv4uogby/AN25BiCsNh0o3rnnA9dYmNQ?dl=0&e=1&preview=log.txt&rlkey=ufc80pc2spzc98cn4atrh26jl">pretraining log</a></td>
  </tr>
  <tr>
    <td>ViT-s</td>
    <th> 96.76 </th>
    <td><a href="https://www.dropbox.com/scl/fo/z1w40ypwbsyqlkywevm5t/AFAnxqw0VPnRmf8c1KoulKU?dl=0&e=1&preview=checkpoint.pth&rlkey=vq3xq6dj4hmtrv1qeah1cnmlg">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/z1w40ypwbsyqlkywevm5t/AFAnxqw0VPnRmf8c1KoulKU?dl=0&e=1&preview=log.txt&rlkey=vq3xq6dj4hmtrv1qeah1cnmlg">pretraining log</a></td>
  </tr>

  <tr>
    <td>Vim-ti</td>
    <th> 95.81 </th>
    <td><a href="https://www.dropbox.com/scl/fo/4q86hsyhxqf0s30sznsi7/AD3K7kL0D9tMCEw2s6GzOGs?dl=0&e=1&preview=checkpoint.pth&rlkey=57wabu98dei6x60u6dxhe33vg">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/4q86hsyhxqf0s30sznsi7/AD3K7kL0D9tMCEw2s6GzOGs?dl=0&e=1&preview=log.txt&rlkey=57wabu98dei6x60u6dxhe33vg">pretraining log</a></td>
  </tr>

  <tr>
    <td>Vim-ti-plus</td>
    <th> 97.39 </th>
    <td><a href="https://www.dropbox.com/scl/fo/93486j0plk4zz185ncmio/AEeLYVl1Cv92ucHYnc5zAXc?dl=0&e=1&preview=checkpoint.pth&rlkey=2scjj7ekkceii4iepexg0huvx">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/93486j0plk4zz185ncmio/AEeLYVl1Cv92ucHYnc5zAXc?dl=0&e=1&preview=log.txt&rlkey=2scjj7ekkceii4iepexg0huvx">pretraining log</a></td>
  </tr>

  <tr>
    <td>Vim-s</td>
    <th> 98.85 </th>
    <td><a href="https://www.dropbox.com/scl/fo/itlxf4cqyvxrbp7kxh43t/ADEuaFPA4Fv5Le96B2T1YZk?dl=0&e=1&preview=checkpoint.pth&rlkey=tf9du6jleuvymfcbhsi67iuf2">checkpoints</a></td>
    <td><a href="https://www.dropbox.com/scl/fo/itlxf4cqyvxrbp7kxh43t/ADEuaFPA4Fv5Le96B2T1YZk?dl=0&e=1&preview=log.txt&rlkey=tf9du6jleuvymfcbhsi67iuf2">pretraining log</a></td>
  </tr>

</table>

## Citation
If you find this repository useful, please consider giving a star and citation (arxiv preprint):
```
@article{nasiri2024vim4path,
  title={Vim4Path: Self-Supervised Vision Mamba for Histopathology Images},
  author={Nasiri-Sarvi, Ali and Trinh, Vincent Quoc-Huy and Rivaz, Hassan and Hosseini, Mahdi S},
  journal={arXiv preprint arXiv:2404.13222},
  year={2024}
}
```
