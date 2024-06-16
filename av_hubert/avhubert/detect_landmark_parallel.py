import multiprocessing as mp
import sys,os,pickle,math
import cv2,dlib,time
import numpy as np
from tqdm import tqdm
import argparse
import skvideo
import skvideo.io
from itertools import repeat
import glob

def load_video(path):
    try:
        videogen = skvideo.io.vread(path)
    except:
        print('***path***',path)
    frames = np.array([frame for frame in videogen])
    return frames

def detect_landmark(image, detector, cnn_detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        rects = cnn_detector(gray)
        rects = [d.rect for d in rects]
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# def detect_face_landmarks(detector, predictor, cnn_detector, input_dir, output_dir, file_ids): 
#     for fid in tqdm(file_ids):
#         base_id = os.path.basename(fid)
#         out_path = os.path.join(output_dir, f"{base_id}.pkl")
#         video_path = os.path.join(input_dir, f"{fid}.mp4")
#         frames = load_video(video_path)
#         landmarks = []
#         for frame in frames:
#             landmark = detect_landmark(frame, detector, cnn_detector, predictor)
#             landmarks.append(landmark)
#         pickle.dump(landmarks, open(out_path, 'wb'))
#     return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detecting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, help='root dir')
    parser.add_argument('--landmark', type=str, help='landmark dir')
    parser.add_argument('--manifest', type=str, help='a list of filenames')
    parser.add_argument('--cnn_detector', type=str, help='path to cnn detector (download and unzip from: http://dlib.net/files/mmod_human_face_detector.dat.bz2)')
    parser.add_argument('--face_predictor', type=str, help='path to face predictor (download and unzip from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    args = parser.parse_args()
    
    skvideo.setFFmpegPath(os.path.dirname(args.ffmpeg))
    print(skvideo.getFFmpegPath())

    # Parallelism
    # detector = dlib.get_frontal_face_detector()
    # cnn_detector = dlib.cnn_face_detection_model_v1(args.cnn_detector)
    # predictor = dlib.shape_predictor(args.face_predictor)

    input_dir = args.root #
    output_dir = args.landmark #
    os.makedirs(output_dir, exist_ok=True)

    fids = [ln.strip() for ln in open(args.manifest).readlines()]
    num_per_shard = math.ceil(len(fids)/args.nshard)
    ranks = range(0, args.nshard)
    files = []
    for rank in ranks:
        start_id = rank*num_per_shard
        end_id = (rank + 1)*num_per_shard
        files.append(fids[start_id: end_id])

    def detect_face_landmarks(file_ids): 
        
        detector = dlib.get_frontal_face_detector()
        cnn_detector = dlib.cnn_face_detection_model_v1(args.cnn_detector)
        predictor = dlib.shape_predictor(args.face_predictor)

        donevideos = glob.glob(os.path.join(output_dir, 'trainval/*/*.pkl'))
        #print('donevideos',donevideos[:3])

        for fid in tqdm(file_ids):
            base_id = os.path.basename(fid)
            #out_path = os.path.join(output_dir, f"{base_id}.pkl")
            video_path = os.path.join(input_dir, f"{fid}.mp4")

            op = os.path.join(output_dir, f"{fid}.pkl")
            if op in donevideos:
                continue

            frames = load_video(video_path)
            landmarks = []
            for frame in frames:
                landmark = detect_landmark(frame, detector, cnn_detector, predictor)
                landmarks.append(landmark)
            pickle.dump(landmarks, open(op, 'wb'))
        return
    
    with mp.Pool(args.nshard) as p:
        p.map(detect_face_landmarks, files)
        
    # detect_face_landmarks(args.face_predictor, args.cnn_detector, args.root, args.landmark, args.manifest, args.rank, args.nshard)i
