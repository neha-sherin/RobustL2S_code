#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:2
#SBATCH -w gnode045
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --output='runout.txt'

python learn_kmeans.py /ssd_scratch/cvit/neha/chemistry/dataset/chem_vidfeatures  train 1 /ssd_scratch/cvit/neha/chemistry/dataset/km.bin 500 --percent 0.6



