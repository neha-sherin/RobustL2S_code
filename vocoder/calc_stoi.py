from pystoi.stoi import stoi
from scipy.io import wavfile
import glob

stoi_val=0
estoi_val=0

path = 'generations_l1chemmaskedfromscratch_60k_audiovocoder75k'		#0.539, 0.354

files = sorted(glob.glob(path+'/*.wav'))
for f in files[:]:
	if '_gt.wav' in f:
		files.remove(f)

for i in range(0, len(files)):
	degfile = files[i]
	gtfile = degfile[:-4]+'_gt.wav'
	
	rate_ref, ref = wavfile.read(gtfile)
	rate_deg, deg = wavfile.read(degfile)
		
	st = stoi( ref, deg, rate, extended=False)	
	est = stoi( ref, deg, rate, extended=True)
	stoi_val += st
	estoi_val += est

print('num files = ', len(files))
print('STOI = ',stoi_val/len(files))
print('ESTOI = ',estoi_val/len(files))
