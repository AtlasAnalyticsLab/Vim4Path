#!/bin/bash
#SBATCH -J CHUM # job name
#SBATCH -w virya1
#SBATCH -n8 # of CPU cores
#SBATCH --mem=30GB # memory reserved (mandatory)

source /etc/profile.d/modules.sh # adding module binaries

module load anaconda/3.2023.03
python extract_patches.py