# make video to 25fps, make audio single channel and sr 16kHz
import subprocess
import os
import glob
import tqdm

videos = glob.glob("/ssd_scratch/cvit/neha/chemistry/vidandaud_ds/*/*.mp4")

template = "ffmpeg -loglevel error -i {} -r 25  -ar 16000 -ac 1 {}"
outdir = '/ssd_scratch/cvit/neha/chemistry/trainval/'
os.makedirs(outdir, exist_ok=True)

#donefiles = os.listdir('/ssd_scratch/cvit/neha/TCD-TIMIT/speaker3/trainval/')

for vidp in tqdm.tqdm(videos):
    folname = os.path.join(outdir, vidp.split('/')[-2])
    filename = vidp.split('/')[-1][:-4]
    os.makedirs(folname, exist_ok=True)

    command2 = template.format(vidp, os.path.join(folname, filename)+".mp4")
    subprocess.call(command2, shell=True)

    #audio_path = os.path.join(folname, filename)+".wav"
    #command1 = f"ffmpeg -i {vidp} -vn -ar 16000 -ac 1 {audio_path}"

    # Execute the ffmpeg command
    #subprocess.call(command1, shell=True)

    #if f.split('/')[-1] not in donefiles:
        #command = template.format(f, f.replace('straightcam','trainval'), f)
        #subprocess.call(command, shell=True)



for split in ["train","val"]:
    info = open("/ssd_scratch/cvit/neha/chemistry/data_splits/"+split+".txt").readlines()
    fnames = [ i.split("|")[0] for i in info ]
    wnames = [ f"trainval/{fname[:-4]}/{fname[-3:]}\n" for fname in fnames ]
    open("/ssd_scratch/cvit/neha/chemistry/"+split+".txt","w").writelines(wnames)

