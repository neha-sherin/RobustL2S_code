#!/bin/bash
#SBATCH -A research
#SBATCH -c 19
#SBATCH --gres=gpu:2
#SBATCH -w gnode047
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --output='runout.txt'

export PYTHONPATH=/home2/neha.sherin/av_hubert/fairseq:$PYTHONPATH


python create_hubert_wrdfile.py 


