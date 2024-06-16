#!/bin/bash
#SBATCH -A neha.sherin
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --nodelist gnode075
#SBATCH --mail-type=END
#SBATCH -c 10



python train.py -p config/chem_av2h/preprocess.yaml -m config/chem_av2h/model.yaml -t config/chem_av2h/train.yaml

