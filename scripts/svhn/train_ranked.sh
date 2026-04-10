#!/bin/bash
#SBATCH --export=ALL
#SBATCH -A gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -J svhn_ranked
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

cd /home/$USER/scratch/cs587-project

mkdir -p results

ml conda
conda activate ./env

python main.py --dataset=svhn --data-aug=0 --model=PreActResNet18 --method=rank_based
