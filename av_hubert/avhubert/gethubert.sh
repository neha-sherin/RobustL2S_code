#!/bin/bash
#SBATCH --mem-per-cpu=2048
#SBATCH --time=10:00:00
#SBATCH --mincpus=10
#SBATCH -w gnode012
#SBATCH --mail-user=neha.sherin@research.iiit.ac.in
#SBATCH --mail-type=ALL

source /home2/neha.sherin/miniconda3/bin/activate
conda activate py38

# create tsv file - contains list of audio file names and duration...
# for AV-HuBERT, I have file_name, video_path, audio_path, 2 numbers..
python /home2/neha.sherin/fairseq/examples/wav2vec/wav2vec_manifest.py /ssd_scratch/cvit/neha/preprocessed_wav --valid-percent 0 --dest /ssd_scratch/cvit/neha/hubert/tsv/ --ext wav

# get hubert emb from 11th transformer layer, in SR and GSLM papers, its taken from 6th layer
# in AV-HuBERT paper, the clusters are taken from [MFCC,9,12,12,12]th layers for 5 training iterations 
python /home2/neha.sherin/fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /ssd_scratch/cvit/neha/hubert/tsv train /ssd_scratch/cvit/neha/hubert/checkpoints/hubert_base_ls960.pt 11 1 0 /ssd_scratch/cvit/neha/hubert/features/

# clustering into 100 clusters- create labels file
python /home2/neha.sherin/fairseq/examples/hubert/simple_kmeans/dump_km_label.py /ssd_scratch/cvit/neha/hubert/features/ train /ssd_scratch/cvit/neha/hubert/checkpoints/km.bin 1 0 /ssd_scratch/cvit/neha/hubert/labels/

# just parsing into one file
python /home2/neha.sherin/speech-resynthesis/scripts/parse_hubert_codes.py --codes /ssd_scratch/cvit/neha/hubert/labels/train_0_1.km --manifest /ssd_scratch/cvit/neha/hubert/tsv/train.tsv --outdir /ssd_scratch/cvit/neha/hubert/parsed_hubert/



