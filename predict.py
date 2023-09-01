# script to use best oss model for prediction
# and plot histograms


import matplotlib.pyplot as plt
import opensoundscape
import glob
import os
import pandas as pd
import numpy as np
import sklearn
import librosa
import torch
import wandb
import random
from  apply_transfer_function import TransferFunction
from convert_audio_to_bits import convert_audio_to_bits

model = opensoundscape.ml.cnn.load_model('/home/michaela/CV4E/model_states/best.model')
# for test data!!!! 
# in case test data is different sampling rate than training data (might just have to stick to one) yikes. What am I going to do about that? if loop? 
model.preprocessor.pipeline.to_spec.params.window_samples = 1000 # 100 window samples
model.preprocessor.pipeline.to_spec.params.overlap_samples = 900 # 90% overlap, for 2000 Fs this means 900 samples, and 0.05 sec bins
model.preprocessor.pipeline.to_spec.params.fft_size = 2000 # FFT = Fs, 1 Hz bins

# load data 

test_clips = pd.read_csv('/home/michaela/CV4E/labeled_data/CINMS18B_one_hot_clips.csv', index_col=[0,1,2])
test_scores = model.predict(test_clips, num_workers=16,batch_size=128)
test_scores.columns = ['pred_D','pred_A','pred_B']
test_all = test_clips.join(test_scores)
test_evaluation = test_all.reset_index()

# D call test
D_eval_index = test_evaluation.index[test_evaluation['D']==1]
D_eval = test_evaluation.loc[D_eval_index]
D_noise_index = test_evaluation.index[test_evaluation['D']==0]
D_noise = test_evaluation.loc[D_noise_index]

plt.hist(D_noise['pred_D'],alpha=0.5,edgecolor='black',color='blue',label='Noise prediction score')
plt.hist(D_eval['pred_D'],alpha=0.5,edgecolor='black',color='orange',label='D call prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('D call prediction scores test')
plt.legend(loc='upper right') # this is progress. look at those high scoring missed detections
# blue is all of the examples in the D call column that did not actually contain a D call. 

# B call test
B_eval_index = test_evaluation.index[test_evaluation['B NE Pacific']==1]
B_eval = test_evaluation.loc[B_eval_index]
B_noise_index = test_evaluation.index[test_evaluation['B NE Pacific']==0]
B_noise = test_evaluation.loc[B_noise_index]

plt.hist(B_noise['pred_B'],alpha=0.5,edgecolor='black',color='blue',label='Noise prediction score')
plt.hist(B_eval['pred_B'],alpha=0.5,edgecolor='black',color='orange',label='B call prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('B call prediction scores test')
plt.legend(loc='upper right') 

# A call test
A_eval_index = test_evaluation.index[test_evaluation['A NE Pacific']==1]
A_eval = test_evaluation.loc[A_eval_index]
A_noise_index = test_evaluation.index[test_evaluation['A NE Pacific']==0]
A_noise = test_evaluation.loc[A_noise_index]

plt.hist(A_noise['pred_A'],edgecolor='black',alpha=0.5,color='blue',label='Noise prediction score')
plt.hist(A_eval['pred_A'],edgecolor='black',alpha=0.5,color='orange',label='A call prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('A call prediction scores validate')
plt.legend(loc='upper right')