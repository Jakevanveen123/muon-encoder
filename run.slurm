#!/bin/bash
#SBATCH --job-name=muon-enc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# usage: sbatch run.slurm --size 17m --treatment muon --lr 0.02 --seed 0
mkdir -p logs
export WANDB_MODE=offline
python train.py --compile "$@"
