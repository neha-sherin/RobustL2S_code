#!/bin/bash
#SBATCH --mem-per-cpu=2048
#SBATCH --time=10:00:00
#SBATCH --mincpus=10
#SBATCH -w gnode075
#SBATCH --mail-user=neha.sherin@research.iiit.ac.in
#SBATCH --mail-type=ALL

source /home2/neha.sherin/miniconda3/bin/activate
conda activate py38

# export PYTHONPATH=/home2/neha.sherin/fairseq:$PYTHONPATH

#python vocoder/scripts/preprocess.py --srcdir /ssd_scratch/cvit/neha/chemistry/wavs --outdir /ssd_scratch/cvit/neha/chemistry/wavs_16khz

# create tsv file - contains list of audio file names and duration...
python av_hubert/fairseq/examples/wav2vec/wav2vec_manifest.py /ssd_scratch/cvit/neha/chemistry/audio/trainval --valid-percent 0 --dest /ssd_scratch/cvit/neha/chemistry/audiohubert/tsv/ --ext wav

# get hubert emb from 11th transformer layer, in SR and GSLM papers, its taken from 6th layer
python av_hubert/fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /ssd_scratch/cvit/neha/chemistry/audiohubert/tsv train /ssd_scratch/cvit/neha/hubert_ckpts/hubert_base_ls960.pt 6 1 0 /ssd_scratch/cvit/neha/chemistry/audiohubert/features/

# clustering into 100 clusters
python av_hubert/fairseq/examples/hubert/simple_kmeans/dump_km_label.py /ssd_scratch/cvit/neha/chemistry/audiohubert/features/ train /ssd_scratch/cvit/neha/hubert_ckpts/km.bin 1 0 /ssd_scratch/cvit/neha/chemistry/audiohubert/labels/

# just parsing into one file
python vocoder/scripts/parse_hubert_codes.py --codes /ssd_scratch/cvit/neha/chemistry/audiohubert/labels/train_0_1.km --manifest /ssd_scratch/cvit/neha/chemistry/audiohubert/tsv/train.tsv --outdir /ssd_scratch/cvit/neha/chemistry/audiohubert/parsed_hubert/


