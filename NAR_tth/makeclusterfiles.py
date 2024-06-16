import os
import argparse
import glob
import numpy as np
from npy_append_array import NpyAppendArray

parser = argparse.ArgumentParser()

parser.add_argument(
    "--outpath",
    type=str,
    default='/ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat/model_output/features/',
)
args = parser.parse_args()

outpath = args.outpath

files = sorted(glob.glob(outpath+'/trainval/*/*.npy'))

#lenfile = open('/ssd_scratch/cvit/neha/chemistry/chem_finetunedavfeat2audfeat_wctc/model_output_172k/test_0_1.len','w')
lenfile = open(outpath+'/test_0_1.len','w')
featurefile = NpyAppendArray(outpath+'/test_0_1.npy')
tsvfile = open(outpath+'/test.tsv','w')

tsvfile.write('/ssd_scratch/cvit/neha/chemistry/audio/\n')

for f in files:
    tsvfile.write('/'.join(f.split('/')[-3:]).replace('.npy','.wav')+'\t0\n')
    val = np.load(f)
    len_ = len(val)
    lenfile.write(f"{len_}\n")
    featurefile.append(val)

