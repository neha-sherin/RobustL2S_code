import cv2
import tqdm
import glob
import os


out_fps = 50
#fps = 25 # video.get(cv2.CAP_PROP_FPS)

vid_path = '/ssd_scratch/cvit/neha/facestar/female_speaker/dataset/video/'
folders = ['trainval']
for folder in folders:
    vid_path_ = vid_path+folder+'/sess/*.mp4'
    all_vids = glob.glob(vid_path_)

    for vid in tqdm.tqdm(all_vids):

        video = cv2.VideoCapture(vid)

        out_filename = vid.replace('video','50fpsvideo')

        # Create the video writer object
        out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (96, 96))

        # Read the frames and write the doubled FPS video
        while True:
            ret, frame = video.read()

            if not ret:
                break

            # Write each frame twice to the output video
            out.write(frame)
            out.write(frame)

        # Release the video objects
        video.release()
        out.release()

