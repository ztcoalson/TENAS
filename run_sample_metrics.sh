#!/bin/bash
#SBATCH -J tenas
#SBATCH -A sail
#SBATCH -p sail
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 0-12:00:00
#SBATCH --output=./logs/slurm-%j.out

module load cuda/11.1

python sample_metrics.py \
    --data_path ../data \
    --search_space_name darts \
    --dataset cifar10 \
    --note 'should_work' \
    --rand_seed 42 \
    --batch_size 16 \
    --n_sample 100 \