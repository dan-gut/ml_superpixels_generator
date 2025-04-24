#!/bin/bash -l

#SBATCH -J sp_train
#SBATCH -A plgsuperpiksele-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=12:00:00
#SBATCH --mem=128000MB
#SBATCH --gres=gpu:1

conda activate py310
module load monai

for dataset in pneumonia path chest derma oct retina breast blood tissue organa organc organs ; do
  python train_resnet.py --dataset ${dataset} --save_path resnets/resnet18_${dataset}_mnist.pt
done;