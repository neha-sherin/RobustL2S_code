import json
import math
import os
import sys

import numpy as np
from torch.utils.data import Dataset

from utils.tools import pad_1D, pad_2D

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.hubert_path = preprocess_config["path"]["hubert_path"]
        self.hubert_file = preprocess_config["path"]["hubert_file"]
        self.hubert_encoder =  preprocess_config["path"]["encoder"]
        
        self.basename = self.process_meta(filename)

        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]

        # input av feature
        hubert_path = os.path.join(
            #self.preprocessed_path,
            "/ssd_scratch/cvit/neha/chemistry/",
            "chem_vidfeatures/vid_feature_files",
            "{}.npy".format(basename),
        )
        vid_feat = np.load(hubert_path)
 

        # target hubert feature
        audiohubpath = os.path.join( 
             "/ssd_scratch/cvit/neha/chemistry/",
             "audiohubert/features/hubertfeature_files",
             "{}.npy".format(basename) )
        aud_feat = np.load(audiohubpath)

        if abs(len(aud_feat)-2*len(vid_feat))>10:
            print('shape mismatch > 5 for ',basename, "by", len(aud_feat)-2*len(vid_feat))
            print('alignment issue between imput and target. Stopping execution')
            sys.exit()
        if len(aud_feat)<2*len(vid_feat):
            vid_feat = vid_feat[:int(len(aud_feat)/2),:]
            aud_feat = aud_feat[:2*len(vid_feat)]
        elif 2*len(vid_feat)<len(aud_feat):
            aud_feat = aud_feat[:2*len(vid_feat)]


        sample = {
            "id": basename,
            "vid_feat": vid_feat, # phone,
            "aud_feat": aud_feat,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            for line in f.readlines():
                n = line.strip()
                name.append(n)
            return name 

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        vid_feats = [data[idx]["vid_feat"] for idx in idxs]
        aud_feats = [data[idx]["aud_feat"] for idx in idxs]

        # *2 to match in and out dimensions
        vidfeat_lens = np.array([2*vid_feat.shape[0] for vid_feat in vid_feats])
        audfeat_lens = np.array([aud_feat.shape[0] for aud_feat in aud_feats])

        vid_feats = pad_2D(vid_feats)
        aud_feats = pad_2D(aud_feats)

        return (
            ids,
            vid_feats,
            vidfeat_lens,
            max(vidfeat_lens),
            aud_feats,
            audfeat_lens,
            max(audfeat_lens)
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["vid_feat"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class VideoDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.basename = self.process_meta(filepath)

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]

        # input av feature
        hubert_path = os.path.join(
            #self.preprocessed_path,
            "/ssd_scratch/cvit/neha/chemistry/",
            "chem_vidfeatures/vid_feature_files",
            "{}.npy".format(basename),
        )
        vid_feat = np.load(hubert_path)

        return (basename, vid_feat)

    def process_meta(self, filename):
        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:
            name = []
            for i,line in enumerate(f.readlines()):
                n = line.strip()
                name.append(n)
            return name

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        vid_feats = np.array([d[1] for d in data])
        vidfeat_lens = np.array([2*vid_feat.shape[0] for vid_feat in vid_feats])

        #texts = pad_1D(texts)
        #return ids, raw_texts, speakers, texts, text_lens, max(text_lens)
        return ids, vid_feats, vidfeat_lens, max(vidfeat_lens)


