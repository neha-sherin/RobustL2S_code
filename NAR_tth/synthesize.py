#python synthesize.py --source /ssd_scratch/cvit/neha/NAM/nar_preprocessed_data_ljnams_nams/val.txt --restore_step 100000 --mode batch -p config/nam/preprocess.yaml -m config/nam/model.yaml -t config/nam/train.yaml

import re
import argparse
from string import punctuation
import json
import torch
import os
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
import tqdm

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, attention_image_summary
from dataset import VideoDataset as VideoDataset
import scipy
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# python /media/newhd/NAM/parrotTTS_NAR/synthesize.py --source /media/newhd/NAM/nar-preprocessed-data/val.txt --restore_step 50000 --mode batch -p /media/newhd/NAM/parrotTTS_NAR/config/nam/preprocess.yaml -m /media/newhd/NAM/parrotTTS_NAR/config/nam/model.yaml -t /media/newhd/NAM/parrotTTS_NAR/config/nam/train.yaml


def synthesize(model, step, configs, outsavepath, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    #pitch_control, energy_control, duration_control = control_values

    count = 0
    final = []
    times = []

    os.makedirs(outsavepath, exist_ok=True)

    for batch in tqdm.tqdm(batchs):
        count+=1
        batch = to_device(batch, device)
        model.test()
        with torch.no_grad():
            sttime = time.time()
            output = model(
                *(batch)
            )
            times.append(time.time()-sttime)
            #print("decoded length",len(output))

            savefolder = '/'.join((outsavepath+batch[0][0]).split('/')[:-1])
            os.makedirs(savefolder, exist_ok=True)
            np.save(outsavepath+batch[0][0], output.to('cpu').numpy())

            #hubert_map = {}
            #if args.mode == "batch":
                #spk = 'whisper'
                #hubert_map['audio'] = '/ssd_scratch/cvit/neha/LJSpeech-1.1/wavs_16khz/{}.wav'.format(batch[0][0])
                #hubert_map['audio'] = '/ssd_scratch/cvit/neha/vctk/audio/{}.wav'.format(batch[0][0])
                #hubert_map['audio'] = '/ssd_scratch/cvit/neha/baseline_data/whisper/wavs/{}.wav'.format(batch[0][0].replace('_nam','_headset'))

            #output = [str(out) for out in output]
            hubert_lengths = len(output)
            #output = " ".join(output)
            #hubert_map['hubert'] = output
            #hubert_map['duration'] = hubert_lengths/50
            #final.append(hubert_map)
        #print("Average time taken",np.mean(times))

        #with open("/ssd_scratch/cvit/neha/NAM/NAR_namHuBert_whisperHuBert/result/result_"+str(step)+".txt",'w') as f:
        #with open("nam_nam2speech_"+str(step)+".txt",'w') as f:
        #    for l in final:
        #        f.write(str(l)+"\n")
        #f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default='/ssd_scratch/cvit/neha/chemistry/dataset/chem_finetunedavfeat2audfeat/model_output/features/',
    )
    args = parser.parse_args()

    if args.mode == "batch":
        assert args.source is not None and args.text is None

    if args.mode == "single":
        args.source = None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    #vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = VideoDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=dataset.collate_fn,
        )

    synthesize(model, args.restore_step, configs, args.outpath, batchs, None)
