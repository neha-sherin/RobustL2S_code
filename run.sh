#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:1
#SBATCH -w gnode045
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --output='runout0.txt'

echo converting video to 25fps and audio to 16kHz, single channel and create train val files containing only the names of files like - "trainval/folder/001"
# if you have text and want to fine-tune av-hubert, put the text in trainval folder along with video files, in .txt format.
python prepare_chem_data.py

cd av_hubert/avhubert/preparation

echo extracting audio for trainval and test split
python lrs3_prepare.py --lrs3 /ssd_scratch/cvit/neha/chemistry --ffmpeg /opt/ffmpeg-5.0.1/bin/ffmpeg --rank 0 --nshard 1 --step 3

echo generate a list of file ids and corresponding text transcriptions
python lrs3_prepare_notext.py --lrs3 /ssd_scratch/cvit/neha/chemistry --ffmpeg /opt/ffmpeg-5.0.1/bin/ffmpeg --rank 0 --nshard 1 --step 4
# if you have text and want to fine-tune avhunert, run the below command - lrs3_prepare.py instead.
#python lrs3_prepare.py --lrs3 /ssd_scratch/cvit/neha/chemistry --ffmpeg /opt/ffmpeg-5.0.1/bin/ffmpeg --rank 0 --nshard 1 --step 4

echo Detect facial landmarks and crop mouth ROI
# you may run `sh run0.sh`, `sh run1.sh` or the following two python commands
python detect_landmark.py --root /ssd_scratch/cvit/neha/chemistry/ --landmark /ssd_scratch/cvit/neha/chemistry/landmark --manifest /ssd_scratch/cvit/neha/chemistry/file.list --cnn_detector /ssd_scratch/cvit/neha/mmod_human_face_detector.dat --face_predictor /ssd_scratch/cvit/neha/shape_predictor_68_face_landmarks.dat --ffmpeg /opt/ffmpeg-5.0.1/bin/ffmpeg --rank 0 --nshard 1

python align_mouth.py --video-direc /ssd_scratch/cvit/neha/chemistry/ --landmark /ssd_scratch/cvit/neha/chemistry/landmark --filename-path /ssd_scratch/cvit/neha/chemistry/file.list --save-direc /ssd_scratch/cvit/neha/chemistry/video --mean-face 20words_mean_face.npy --ffmpeg /opt/ffmpeg-5.0.1/bin/ffmpeg --rank 0 --nshard 1


python count_frames.py --root /ssd_scratch/cvit/neha/chemistry --manifest /ssd_scratch/cvit/neha/chemistry/file.list --nshard 1 --rank 0

cat /ssd_scratch/cvit/neha/chemistry/nframes.audio.0 > /ssd_scratch/cvit/neha/chemistry/nframes.audio
cat /ssd_scratch/cvit/neha/chemistry/nframes.video.0 > /ssd_scratch/cvit/neha/chemistry/nframes.video

echo create dictionary, file list for fine-tuning av-hubert model
python vidmanifest.py --lrs3  /ssd_scratch/cvit/neha/chemistry
# if you have text and want to fine-tune av-hubert, run below script - lrs3_manifest.py. not vidmanifest.py
#python lrs3_manifest.py --lrs3  /ssd_scratch/cvit/neha/chemistry   --valid-ids /ssd_scratch/cvit/neha/chemistry/data_split/val.txt --vocab-size 1000

cd ..

# if you want to fine-tune av-hubert, you need corresponding text and in the above steps, do 
# .wrd files will be created in 30h_data folder. This is task.label_dir.
# download base model from online
echo FINETUNE
fairseq-hydra-train --config-dir ./conf/pretrain/ --config-name base_vox_iter5.yaml task.data=/ssd_scratch/cvit/neha/chemistry/30h_data/ task.label_dir=/ssd_scratch/cvit/neha/chemistry/30h_data model.label_rate=25  hydra.run.dir=`pwd` common.user_dir=`pwd`

echo EXTRACT VIDEO FEATURES
cd clustering
# mention any checkpoint that u want to use to extract video-features
# download av-hubert ckpt from online if not finetuned
python dump_vidhubfeat.py /ssd_scratch/cvit/neha/chemistry/30h_data/ train /ssd_scratch/cvit/neha/avhubert_ckpt/pretrain/base_vox_433h.pt  12 1 0 /ssd_scratch/cvit/neha/chemistry/chem_vidfeatures --user_dir `pwd`/../

#echo if training vocoder with just video-features:
#echo learn kmeans and extract cluster ids
#python learn_kmeans.py /ssd_scratch/cvit/neha/chemistry/chem_vidfeatures/  train 1 /ssd_scratch/cvit/neha/chemistry/avhubert_kmeans_model/km.bin 500 --percent 0.6

#python dump_km_label.py /ssd_scratch/cvit/neha/chemistry/avhubert/features/ train /ssd_scratch/cvit/neha/chemistry/avhubert_kmeans_model/km.bin 1 0 /ssd_scratch/cvit/neha/avhubert/labels/

#cd ..

#python parse_hubert_codes.py --codes /ssd_scratch/cvit/neha/chemistry/avhubert/labels/train_0_1.km --manifest /ssd_scratch/cvit/neha/chemistry/30h_data/train.tsv --outdir /ssd_scratch/cvit/neha/chemistry/avhubert/parsed_hubert/

cd ../../..

#echo now train the vocoder
#sh vocoder/run.sh   # set config file to `hubert500_lut.json` if using 500 clusters of video

echo EXTRACT AUDIO SSL
# change gnode in run.sh file, or just run its commans.
sh gethubert.sh

echo TRAIN SEQ2SEQ MODEL
cd NAR_tth
sh run.sh 

echo synthesize output
python synthesize.py --source /ssd_scratch/cvit/neha/chemistry/val.txt --restore_step 10000 --mode batch -p config/chem_av2h/preprocess.yaml -m config/chem_av2h/model.yaml -t config/av2h/train.yaml --outpath /ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat/model_output/features/

# change the paths in the file
python makeclusterfiles.py --outpath /ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat/model_output/features

cd ..

# clustering into 100 clusters
python av_hubert/fairseq/examples/hubert/simple_kmeans/dump_km_label.py /ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat/model_output/features/ test /ssd_scratch/cvit/neha/hubert_ckpts/km.bin 1 0 /ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat/model_output/labels/

# just parsing into one file
python NAR_tth/parse_hubert_codes.py --codes /ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat/model_output/labels/test_0_1.km --manifest /ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat/model_output/features/test.tsv --outdir /ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat/model_output/parsed_hubert/

echo TRAIN VOCODER

sh vocoder/run.sh

echo INFER RESULTS
cd vocoder

python inference.py --checkpoint_file /ssd_scratch/cvit/neha/chemistry/vocoder_ckpt  --output_dir generations --input_code_file  /ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat/model_output/parsed_hubert/train.txt 


