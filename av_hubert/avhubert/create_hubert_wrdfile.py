import tqdm


with open('/ssd_scratch/cvit/neha/lrs2/30h_data/train.tsv','r') as f:
    list_ = f.readlines()

with open('/ssd_scratch/cvit/neha/lrsaudio_huberts/parsed_hubert/train.txt','r') as f2:
    hublines = f2.readlines()
    
with open('/ssd_scratch/cvit/neha/lrsaudio_huberts/parsed_hubert/val.txt','r') as f2:
    hublines.extend(f2.readlines())
    
with open('/ssd_scratch/cvit/neha/lrsaudio_huberts/parsed_hubert/test.txt','r') as f2:
    hublines.extend(f2.readlines())

with open('/ssd_scratch/cvit/neha/train_hubert.wrd','w') as trainwrd:
    for lines in tqdm.tqdm(list_[1:]):
        audiopath = lines.split('\t')[2]
        # print(audiopath)
        for hubsinfo in hublines:
            hubline = eval(hubsinfo)
            if hubline['audio']==audiopath:
                trainwrd.write(hubline['hubert']+'\n')
                break
