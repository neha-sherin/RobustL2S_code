#!/bin/bash
#SBATCH -A research
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -w gnode047      #34
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --output='runout.txt'

export PYTHONPATH=/home2/neha.sherin/av_hubert/fairseq:$PYTHONPATH


fairseq-hydra-train --config-dir ./conf/pretrain/ --config-name base_vox_iter5.yaml task.data=/ssd_scratch/cvit/neha/chemistry/dataset/30h_data/ task.label_dir=/ssd_scratch/cvit/neha/50fpsfacestar/audio_hub_labels  model.label_rate=100  hydra.run.dir=`pwd` common.user_dir=`pwd`


#fairseq-hydra-train --config-dir ./conf/finetune/ --config-name base_lrs3_433h.yaml  task.data=/ssd_scratch/cvit/neha/lrs2/433h_data/ task.label_dir=/ssd_scratch/cvit/neha/lrs2/433h_data/ task.tokenizer_bpe_model=/ssd_scratch/cvit/neha/lrs2/spm1000/spm_unigram1000.model model.w2v_path=/ssd_scratch/cvit/neha/avhubert_ckpt/pretrain/base_vox_iter5.pt   hydra.run.dir=`pwd` common.user_dir=`pwd`

#fairseq-hydra-train --config-dir ./conf/finetune/ --config-name base_lrs3_30h.yaml  task.data=/ssd_scratch/cvit/neha/lrs2/30h_data/ task.label_dir=/ssd_scratch/cvit/neha/lrs2/30h_data/ task.tokenizer_bpe_model=/ssd_scratch/cvit/neha/lrs2/spm1000/spm_unigram1000.model model.w2v_path=/ssd_scratch/cvit/neha/avhubert_ckpt/pretrain/base_vox_iter5.pt   hydra.run.dir=`pwd` common.user_dir=`pwd`

