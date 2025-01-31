#!/bin/bash
#SBATCH -J tenas
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 0-12:00:00
#SBATCH --output=./logs/slurm-%j.out

module load cuda/11.1

python prune_launch.py \
    --space darts \
    --dataset cifar10 \
    --gpu 0 \
    --seed 48483 \
    --note noise-50%-diff-denoise-4 \
    --poisons_type diffusion_denoise \
    --poisons_path "/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/diffusion_denoise/datasets/denoised/gc_cifar10/denoise_gaussian_noise/denoised_w_sigma_0.1.pt" \