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
from keras.constraints import Constraint
from keras.initializers import Constant

fs=16000;

def rc2AR_Model(N,rc):
	a0=Lambda(lambda x: tf.expand_dims(x[:,0],axis=-1))(rc);

	for j in range(1,N):
		rc1=Lambda(lambda x: tf.expand_dims(x[:,j],axis=-1))(rc);
		ra=Lambda(lambda x: x[1]*tf.keras.backend.reverse(x[0],axes=-1)+x[0])([a0,rc1]);
		a0=Lambda(lambda x: tf.concat([x[0],x[1]],axis=1))([ra,rc1]);
	a0=Lambda(lambda x: tf.expand_dims(x,axis=-1))(a0);
	return(a0);

def GetFrames(wn,Nw,ImpHistory,bsz,Ns):
	wnfrms=np.zeros((1,bsz,Nw+ImpHistory));
	myrange=np.arange(ImpHistory,wn.shape[0]-Nw+Ns,Ns);
	if(myrange.shape[0]!=bsz):
		raise ValueError("Oops!  Batch size and length doesnot match")
	for idx,j in enumerate(myrange):
		wnfrms[0,idx,:]=wn[j-ImpHistory:j+Nw];
	return wnfrms;

class DataGenerator(Sequence):
	def __init__(self,filename,Nw,tr_flg=1,bsz=32,ImpHistory=200,Ns=160):
		self.bsz=bsz;
		self.Ns=Ns;
		self.ImpHistory=ImpHistory;
		self.Nw=Nw;
		self.files=np.genfromtxt(filename,dtype='str');
		self.tr_flg=tr_flg
		if(tr_flg==0):
		    self.files=self.files[100000:120000];
		elif(tr_flg==1):
		    self.files=self.files[0:100000];
		else:
		    self.files=self.files;		

		self.L=len(self.files);
		shuffle(self.files)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		data=scipy.io.loadmat('../../'+self.files[idx])
		rcinpt=data['mel_spec'];
		rcinpt=(rcinpt-15.0)/15.0;
		f0=data['pt']/float(fs);
		ps=data['pv'];
		f0_exc=data['exc_frms'][np.newaxis,:,:];
		#if(self.tr_flg==2):
		w_exc=np.random.randn(rcinpt.shape[0]*self.Ns+self.ImpHistory+self.Nw-self.Ns)
		w_exc=GetFrames(w_exc,self.Nw,self.ImpHistory,rcinpt.shape[0],self.Ns);
		mag_spec=data['mag_spec'];
		feat=np.concatenate((rcinpt,f0,ps),axis=1)[np.newaxis,:,:];	
		if(np.sum(np.isfinite(feat))!=np.prod(feat.shape)):
			raise ValueError("invalid value found in feat")
		if(np.sum(np.isfinite(mag_spec))!=np.prod(mag_spec.shape)):
			raise ValueError("invalid value found in mage spec")

		if(self.tr_flg==2):
			return([feat,f0_exc,w_exc],[data['x'],0.01*mag_spec[np.newaxis,2:-2,:]]);
		else:
			return([feat,f0_exc,w_exc],0.01*mag_spec[np.newaxis,2:-2,:]);#0.01*

	def on_epoch_end(self):#
		shuffle(self.files)

def GetModel(Ni,Nn,Ns,Nw,hann_win,mode='train',ImpHistory=200,nUnits=128,fil_type='gamma'):
	if(mode=='train'):
		featinp = Input(shape=(32, 21))
		Pitch_exc=Input(shape=(32,Nw+ImpHistory));
		wn_exc=Input(shape=(32,Nw+ImpHistory));
	else:
		featinp = Input(shape=(None, 21))
		Pitch_exc=Input(shape=(None,Nw+ImpHistory));
		wn_exc=Input(shape=(None,Nw+ImpHistory));

	
	padding='same'
	ival=1e-3;
	fconv1 = Conv1D(nUnits, 3, padding=padding, activation='tanh', name='feature_conv1')(featinp)
	fconv2 = Conv1D(nUnits, 3, padding=padding, activation='tanh', name='feature_conv2')(fconv1)

	fdense1 = TimeDistributed(Dense(nUnits, activation='tanh', name='feature_dense1'))(fconv2)
	fdense2 = TimeDistributed(Dense(nUnits, activation='tanh', name='feature_dense2'))(fdense1)

	rcout=TimeDistributed(Dense(Ni,activation='tanh',kernel_initializer=Constant(1e-6)))(fdense2);
	soe_out=TimeDistributed(Dense(1,activation='linear'))(fdense2);
	soe_out=TimeDistributed(Lambda(lambda x: tf.exp(x)))(soe_out);
	rcnout=TimeDistributed(Dense(Nn,activation='linear'))(fdense2);
	rcnout=TimeDistributed(Lambda(lambda x: tf.exp(x)))(rcnout);

	rcout=Lambda(lambda x: x[0,:,:])(rcout);
	a_i=rc2AR_Model(Ni,rcout);
	a_i=Lambda(lambda x: x[:,:,0],name='ai_out')(a_i)
	############ Synthesis##########
	sp=TwoTVLpFilter_Batch_FB(Ni,Nn,Pitch_exc,rcnout,a_i,soe_out,wn_exc,fil_type);
	sp_win=Lambda(lambda x: tf.multiply(x[:,ImpHistory:,0],tf.constant(hann_win)))(sp);
	sp_syn=Lambda(lambda x: overlap_and_add(x,Ns))(sp_win);
	sp_frms=Lambda(lambda x: frame(x,Nw,Ns))(sp_syn);
	sp_frmswin=Lambda(lambda x: tf.multiply(x,tf.constant(hann_win)))(sp_frms);
	sp_spec=Lambda(lambda x:tf.math.real(tf.abs(fft(tf.cast(x,'complex64')))[:,:161]),name='spec_out')(sp_frmswin)
	sp_spec=Lambda(lambda x: tf.expand_dims(sp_spec[2:-2,:],axis=0))(sp_spec);
	if(mode=='syn'):
		model=Model([featinp,Pitch_exc,wn_exc],[sp_syn,a_i,rcnout,soe_out]);
	else:
		model=Model([featinp,Pitch_exc,wn_exc],sp_spec);
	model.summary();
	return model

def gt_fil(shape,dtype):
	data=scipy.io.loadmat('gtfb_60len_40ord.mat');
	arr=data['gt'].astype('float32');
	arr=arr[:,np.newaxis,:];
	return(tf.constant(arr));

class UnitGainConstraintFB(Constraint):
    def __init__(self):
        print('unit');

    def __call__(self, w):
        new_w = w-tf.reduce_mean(w,axis=0,keepdims=True);
        return new_w

class UnitGainConstraint(Constraint):
    def __init__(self):
        print('unit');

    def __call__(self, w):
        new_w = w/tf.reduce_sum(w);
        return new_w

def TwoTVLpFilter_Batch_FB(p1,p2,z,an,ai,soe,wn,fil_type):
	
	
	class ARfilter(Layer):
		def __init__(self, nch,order,**kwargs):
			self.output_size = nch
			self.state_size = order
			super(ARfilter, self).__init__(**kwargs)

		def call(self, inputs, states,constants):
			prev_output=states[0]	
			aa=constants[0]
			hn = -K.batch_dot(prev_output, aa)+inputs;
			output=tf.concat([hn,prev_output[:,:-1]],axis=-1);#
			return hn, [output]

	an=Lambda(lambda x: x[0,:,:])(an);
	soe=Lambda(lambda x: x[0,:,:])(soe);
	z=Lambda(lambda x: tf.expand_dims(x[0,:,:],axis=-1))(z);
	wn=Lambda(lambda x: tf.expand_dims(x[0,:,:],axis=-1))(wn);	
	if(fil_type=='gamma'):
		wn_sigma_bank=Conv1D(p2,60,padding='same',name='fb_noise',use_bias=False,kernel_initializer=gt_fil,trainable=False)(wn);
	else:
		wn_sigma_bank=Conv1D(p2,60,padding='same',name='fb_noise',use_bias=False)(wn);
		
	wn_sigma_ap=Lambda(lambda x: tf.reduce_mean(tf.multiply(x[0],tf.expand_dims(x[1],axis=1)),keepdims=True,axis=-1),name='noise_exc')([wn_sigma_bank,an]);				
	x_imp=Lambda(lambda x: tf.multiply(x[0],tf.expand_dims(x[1],axis=-1)))([z,soe]);	
	x_imp=Conv1D(1,31,kernel_constraint=UnitGainConstraint(),padding='same',use_bias=False,name='exc_fil')(x_imp);
	
	exc=Lambda(lambda x: x[0]+x[1])([wn_sigma_ap,x_imp]);
	with tf.device('/CPU:0'):
		xfil = RNN(ARfilter(1,p1),return_sequences=True,unroll=True)(exc,constants=[ai])

	return(xfil);
