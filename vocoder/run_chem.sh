#!/bin/bash
#SBATCH -A research
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 10
#SBATCH --time=4-00:00:00
#SBATCH --nodelist gnode070 #50     #38   #34
#SBATCH --mail-type=END


module add u18/cudnn/8.4.0-cuda-11.6 u18/cuda/11.6


python train.py --checkpoint_path /ssd_scratch/cvit/neha/chemaudio_sr_vocoder/ --config configs/chem/hubert100_lut.json --checkpoint_interval 5000

#python train.py --checkpoint_path /ssd_scratch/cvit/neha/chem_avfeat2audfeat/chem_sr_vocoder/ --config configs/chem/hubert100_lut_out.json --checkpoint_interval 5000


