#!/bin/bash
#SBATCH -J V_C # job name
#SBATCH -w virya2
#SBATCH -n16 # of CPU cores
#SBATCH --mem=80GB # memory reserved (mandatory)
#SBATCH --gpus=4
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.nasiri1998@gmail.com
source /etc/profile.d/modules.sh # adding module binaries

module load anaconda/3.2023.03
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25678 main_dino_vim.py --data_path /home/a_n29343/CHUM/VIM4Path/datasets/CHUM/output_vim/ --output_dir checkpoints/checkpoints1280  --batch_size_per_gpu 15  --num_workers 4