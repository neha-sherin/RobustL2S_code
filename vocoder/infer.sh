#!/bin/bash
#SBATCH -A neha.sherin
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 10
#SBATCH --time=3-00:00:00
#SBATCH --nodelist gnode057     #50    #46   #34
#SBATCH --mail-type=END


module add u18/cudnn/8.4.0-cuda-11.6 u18/cuda/11.6

python inference.py --checkpoint_file /ssd_scratch/cvit/neha/NAM/baseline_data/vctk/vocoder_trimmednam_ckpt/  --output_dir  /ssd_scratch/cvit/neha/NAM/baseline_data/vctk/vctk_namvoice   --input_code_file /ssd_scratch/cvit/neha/NAM/baseline_data/vctk/allvctk.txt

#python inference.py --checkpoint_file /ssd_scratch/cvit/neha/speechresynth_vq_chkpoints/  --output_dir /ssd_scratch/cvit/neha/vctk_generated_ljoutput_nam2speechdir   --input_code_file ../parrotTTS_NAR/vctk_nam2lj_100000.txt 

#python inference.py --checkpoint_file /ssd_scratch/cvit/neha/speechresynth_vq_chkpoints/  --output_dir /ssd_scratch/cvit/neha/vctk_generated_ljoutput   --input_code_file ../nam/result_vctk_240000.txt



#python inference.py --checkpoint_file /ssd_scratch/cvit/neha/NAM/vocoder_ckpt/trimnam_checkpoints/  --output_dir /ssd_scratch/cvit/neha/vctk/generations_vctknam_nosil --input_code_file /ssd_scratch/cvit/neha/vctk/hubert/parsed_hubert/vctkdata.txt

#python inference.py --checkpoint_file /ssd_scratch/cvit/neha/NAM/vocoder_ckpt/trimnam_checkpoints/  --output_dir /ssd_scratch/cvit/neha/vctk/generations_vctknam --input_code_file /ssd_scratch/cvit/neha/vctk/preprocessed/train.txt



#python train.py --checkpoint_path /ssd_scratch/cvit/neha/srcheckpoints/facestar_vid_avhubert --config configs/facestar_hubert/hubert500_lut.json --checkpoint_interval 2000

#python -m torch.distributed.launch --nproc_per_node 2 train.py --checkpoint_path /ssd_scratch/cvit/neha/srcheckpoints/facestar_2000 --config configs/facestar_orig/hubert2000_lut.json --checkpoint_interval 2000

#python -m torch.distributed.launch --nproc_per_node 2 train.py --checkpoint_path /ssd_scratch/cvit/neha/srcheckpoints/facestar --config configs/facestar_orig/hubert100_lut.json --checkpoint_interval 2000

#python train.py --checkpoint_path /ssd_scratch/cvit/neha/srcheckpoints/facestar --config configs/facestar/hubert100_lut.json --checkpoint_interval 2000

#python train.py --checkpoint_path /ssd_scratch/cvit/neha/srcheckpoints/ljsp --config configs/LJSpeech/hubert100_lut.json --checkpoint_interval 5000

#python train.py --checkpoint_path /ssd_scratch/cvit/neha/correct_srckpt_wnoise_wof0 --config configs/vctkiemo/hubert100_lut.json --checkpoint_interval 5000

#python -m torch.distributed.launch --nproc_per_node 4 train.py --checkpoint_path /scratch/skosgi242/checkpoints/vq --config configs/LJSpeech/hubert100_lut.json
