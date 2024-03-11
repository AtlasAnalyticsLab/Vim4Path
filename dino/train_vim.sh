#!/bin/bash
#SBATCH -J V_C # job name
#SBATCH -w virya2
#SBATCH -n16 # of CPU cores
#SBATCH --mem=120GB # memory reserved (mandatory)
#SBATCH --gpus=4
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.nasiri1998@gmail.com
source /etc/profile.d/modules.sh # adding module binaries

module load anaconda/3.2023.03
# OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 main.py --data_path /home/a_n29343/VIM4Path/datasets/Camelyon16/224_5x/ --output_dir checkpoints/camelyon16_224_5x/vim-s_224-96_2 --image_size 224 --image_size_down 96  --num_workers 4 --batch_size_per_gpu 64 --arch vim-s
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29800 main.py --data_path /home/a_n29343/VIM4Path/datasets/Camelyon16/512_5x/ --output_dir checkpoints/camelyon16_512_5x/vim-s-24_512_384/ --image_size 512 --image_size_down 384  --num_workers 4 --batch_size_per_gpu 13 --arch vim-s 
