#!/bin/bash
#SBATCH -J V_C # job name
#SBATCH -w virya1
#SBATCH -n16 # of CPU cores
#SBATCH --mem=60GB # memory reserved (mandatory)
#SBATCH --gpus=4

source /etc/profile.d/modules.sh # adding module binaries

module load anaconda/3.2023.03
python  -m torch.distributed.launch --nproc_per_node=4 main_dino_vim.py --data_path /home/a_n29343/CHUM/VIM4Path/datasets/CHUM/output_vim/ --output_dir checkpoints --batch_size_per_gpu 15