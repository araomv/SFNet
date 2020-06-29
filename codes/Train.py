import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from scipy.signal import lfilter
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
from ulaw import ulaw2lin, lin2ulaw
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
model=GetModel(Ni,Nn,Ns,Nw,hann_win,nUnits=32,fil_type='learn');
print('..model created');
########loss function###############
num_mels=80;
M=librosa.filters.mel(fs, Nw, n_mels=num_mels).transpose();
M=M[np.newaxis,:,:]
print(M.shape)
M=tf.constant(M);
epsilon=1e-7;

def mse_melspec(y_true, y_pred):
	#y_true=y_true[0,:,:]
	y_true=K.batch_dot(y_true,M);
	y_pred=K.batch_dot(y_pred,M);
	first_log = K.log(K.clip(y_pred, epsilon, None) )
	second_log = K.log(K.clip(y_true, epsilon, None) )
	return K.mean(K.square(first_log - second_log))

def mae_melspec(y_true, y_pred):
	#y_true=y_true[0,:,:]
	y_true=K.batch_dot(y_true,M);
	y_pred=K.batch_dot(y_pred,M);
	first_log = K.log(K.clip(y_pred, epsilon, None) )
	second_log = K.log(K.clip(y_true,epsilon, None) )
	return K.mean(K.abs(first_log - second_log))


def mse_log_spec(y_true, y_pred):
	#y_true=y_true[0,:,:]
	first_log = K.log(K.clip(y_pred, K.epsilon(), None) )
	second_log = K.log(K.clip(y_true, K.epsilon(), None) )
	return K.mean(K.square(first_log - second_log))



def mae_log_spec(y_true, y_pred):
	#y_true=y_true[0,:,:]
	first_log = K.log(K.clip(y_pred, epsilon, None))
	second_log = K.log(K.clip(y_true, epsilon, None))
	return K.mean(K.abs(first_log - second_log))

def sum_melspec(y_true, y_pred):
	#y_true=y_true[0,:,:]
	y_true=K.batch_dot(y_true,M);
	y_pred=K.batch_dot(y_pred,M);
	first_log = K.log(K.clip(y_pred, epsilon, None) )
	second_log = K.log(K.clip(y_true,epsilon, None) )
	return K.mean(K.abs(first_log - second_log))+K.mean(K.square(first_log - second_log))


run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata= tf.RunMetadata()
########Model training#############
model.compile(optimizer='adam',loss=mae_melspec,metrics=[mse_melspec]);#, 
expts_dir='./expts/'
callback1=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=8,verbose=0, mode='auto')
callback2=keras.callbacks.ModelCheckpoint(expts_dir+'checkpoint-lpnetdata_expvuv_wn_int_moredata_32Un_learnable-{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
print('fitting model')
#model.load_weights('./expts/checkpoint-lpnetdata_expvuv_wn_int-01-0.60.hdf5');
history=model.fit_generator(DataGenerator('./all_tr_files_int',Nw,1),epochs=70,validation_data=DataGenerator('./all_tr_files_int',Nw,0),callbacks=[callback1,callback2], workers=18, use_multiprocessing=True)


