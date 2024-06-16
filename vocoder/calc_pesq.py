from pesq import pesq
from scipy.io import wavfile
import glob

val=0

path = 'generations_eh_18k'

files = sorted(glob.glob(path+'/*.wav'))
for f in files[:]:
        if '_gt.wav' in f:
                files.remove(f)

for i in range(0, len(files)):
        degfile = files[i]
        gtfile = degfile[:-4]+'_gt.wav'

        rate, ref = wavfile.read(gtfile)
        rate, deg = wavfile.read(degfile)

        val += pesq(rate, ref, deg, 'wb')

print('PESQ = ',val/len(files))
