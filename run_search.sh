#!/bin/bash
#SBATCH -J tenas
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 0-12:00:00
#SBATCH --output=./logs/slurm-%j.out

# 1: 42
# 2: 0
# 3: 77
# 4: 100

python prune_launch.py \
    --space darts \
    --dataset cifar10 \
    --gpu 0 \
    --seed 1181 \
    --note noise4-1% \
    --poisons_type clean_label \
    --poisons_path /nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/noise/noise-cifar10-1.0%.pth \