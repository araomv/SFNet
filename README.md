# SFNet
Synthesized samples and code for the paper SFNet: A computationally efficient source filter model based neural speech synthesis

synthesized samples can be found here:https://araomv.github.io/SFNet/

Codes:

Run Tain.py to train the model. It requires following files to be present
  1. all_tr_files_int: list of names (each file containing mel spec, pitch, pitch stregth and the corresponding magnitude spectrum)
  
  
GetModel4.py: It has the implementation of SFNet layers.
custom_op.py, my_utils.op : to support the framing and fft functions in tf.
gtfb_60len_40ord.mat: Mat file containing the gamma tone filter bank for SFNet-U.

