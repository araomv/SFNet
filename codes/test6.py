import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
set_session(tf.Session(config=config))

from keras.layers import *
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Layer
import numpy as np
from keras.models import Model
import scipy.io
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from tensorflow.spectral import fft
import scipy.io.wavfile
from scipy.io import loadmat
from scipy.signal import lfilter, freqz
from keras.utils import Sequence
from random import shuffle
from custom_op import frame
import h5py
import time
from tensorflow.python.client import timeline
import librosa
import librosa.filters
from keras.initializers import Constant
from tensorflow.compat.v1.signal import overlap_and_add
import os
from GetModel4 import GetModel,DataGenerator
import sys
########prameters#################
Ni=25;
Nn=40;
fs=16000;
Nw=round(20e-3*fs)
Ns=round(10e-3*fs)
bsz=32;
hann_win=np.hanning(Nw).astype('float32');
hann_win=hann_win*np.sqrt(Ns/np.sum(hann_win**2));
###############model building######
model=GetModel(Ni,Nn,Ns,Nw,hann_win,mode='syn',nUnits=32,fil_type='learn');
model.load_weights('checkpoint-lpnetdata_expvuv_wn_int_moredata_32Un_learnable-33-0.44.hdf5');
print('..model created');
dgen=DataGenerator('../../testfiles_unseen',Nw,2)
###############data loading#########

files=dgen.files;

for idx in range(len(files)):
	msg = "item %i of %i" % (idx, len(files)-1)
	sys.stdout.write(msg + chr(8) * len(msg))
	path, wavefilename = os.path.split(files[idx])
	try:
		inp,out=dgen.__getitem__(idx)
	except:
		print('skipping '+files[idx])
		continue;
	wavs=model.predict(inp,steps=1);
	scipy.io.savemat('./syn_wavs_unseen/'+wavefilename[:-4]+'.mat',{'org':out[0],'syn':wavs[0],'ai':wavs[1],'an':wavs[2],'soe':wavs[3],'inp':inp[0]});
	sys.stdout.flush()

	#exit(0)
	


