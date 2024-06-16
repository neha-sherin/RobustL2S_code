#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:1
#SBATCH -w gnode045
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --output='runout0.txt'


python align_mouth.py --video-direc /ssd_scratch/cvit/neha/chemistry/dataset --landmark /ssd_scratch/cvit/neha/chemistry/dataset/landmark --filename-path /ssd_scratch/cvit/neha/chemistry/dataset/file.list --save-direc /ssd_scratch/cvit/neha/chemistry/dataset/video --mean-face 20words_mean_face.npy --ffmpeg /opt/ffmpeg-5.0.1/bin/ffmpeg --rank 0 --nshard 2

python detect_landmark.py --root /ssd_scratch/cvit/neha/chemistry/dataset/ --landmark /ssd_scratch/cvit/neha/chemistry/dataset/landmark --manifest /ssd_scratch/cvit/neha/chemistry/dataset/file.list --cnn_detector /ssd_scratch/cvit/neha/mmod_human_face_detector.dat --face_predictor /ssd_scratch/cvit/neha/shape_predictor_68_face_landmarks.dat --ffmpeg /opt/ffmpeg-5.0.1/bin/ffmpeg --rank 0 --nshard 2



#python align_mouth.py --video-direc /ssd_scratch/cvit/neha/lrs2 --landmark /ssd_scratch/cvit/neha/lrs2/landmark --filename-path /ssd_scratch/cvit/neha/lrs2/file.list --save-direc /ssd_scratch/cvit/neha/lrs2/video --mean-face 20words_mean_face.npy --ffmpeg /opt/ffmpeg-5.0.1/bin/ffmpeg --rank 0 --nshard 3


