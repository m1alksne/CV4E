# script to train CNN with opensoundscape package

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

# print(torch.__version__)

# read in train and validation dataframes
train_clips = pd.read_csv('/home/michaela/CV4E/labeled_data/train_balanced_1500.csv', index_col=[0,1,2]) 
validate_clips = pd.read_csv('/home/michaela/CV4E_oss/pre_processing/labeled_data/validate_clips.csv', index_col=[0,1,2])
print(train_clips.sum())
print(validate_clips.sum())

calls_of_interest = ["D", "A NE Pacific", "B NE Pacific"] #define the calls for CNN
model = opensoundscape.CNN('resnet18',classes=calls_of_interest,sample_duration=15.0, single_target=False) # create a CNN object designed to recognize 15-second samples
opensoundscape.ml.cnn.use_resample_loss(model) # loss function for mult-target classification

# moodify model preprocessing for making spectrograms the way I want them
model.preprocessor.pipeline.to_spec.params.window_type = 'hamming' # using hamming window (Triton default)
model.preprocessor.pipeline.to_spec.params.window_samples = 1600 # 1600 window samples
model.preprocessor.pipeline.to_spec.params.overlap_samples = 1400 # 90% overlap, for 3200 Fs this means 1400 samples, and 0.05 sec bins
model.preprocessor.pipeline.to_spec.params.fft_size = 3200 # FFT = Fs, 1 Hz bins
model.preprocessor.pipeline.to_spec.params.decibel_limits = (-200,200) # oss preprocessing sets dB limits. These get reset when tf is applied
model.preprocessor.pipeline.to_spec.params.scaling = 'density'
model.preprocessor.pipeline.bandpass.params.min_f = 10
model.preprocessor.pipeline.bandpass.params.max_f = 150
#model.preprocessor.pipeline.add_noise.bypass=True
model.preprocessor.pipeline.frequency_mask.set(max_width = 0.03, max_masks=10)
model.preprocessor.pipeline.time_mask.set(max_width = 0.1, max_masks=10)
model.preprocessor.pipeline.add_noise.set(std=0.2)
model.preprocessor.pipeline.random_affine.bypass=True
# model.preprocessor.out_shape = [224,448,3] # resize image the size that I want ? might not work with pre-trained weights ?
model.preprocessor.height = 224
model.preprocessor.width = 448
model.preprocessor.channels = 3
model.optimizer_params['lr']=0.001 # learning rate (pretty low but not too low) 
model.lr_cooling_factor = 0.3 # decrease learning rate by multiplying 0.001*0.3 every ten epochs
model.wandb_logging['n_preview_samples']=100 # number of samples that I want to look at 

model.preprocessor.insert_action(
    action_index='convert_to_bits', #give it a name
    action=opensoundscape.preprocess.actions.Action(convert_audio_to_bits), #the action object
    after_key='load_audio') #where to put it (can also use before_key=...)

model.preprocessor.insert_action(
    action_index='apply_tf', #give it a name
    action= TransferFunction(decibel_limits=(40,140)), #the action object
    after_key='to_spec') #where to put it (can also use before_key=...)

wandb_session = wandb.init( #initialize wandb logging 
        entity='BigBlueWhale', #replace with your entity/group name
        project='opensoundscape training BigBlueWhale',
        name='Trial 10: balanced noise, with augmentation')

model.train(
    train_clips, 
    validate_clips, 
    epochs = 10, 
    batch_size= 128, 
    log_interval=1, #log progress every 1 batches
    num_workers = 16, #16 parallelized cpu tasks for preprocessing
    wandb_session=wandb_session,
    save_interval = 1, #save checkpoint every 1 epoch
    save_path = '/home/michaela/CV4E/model_states/' #location to save checkpoints (epochs)
)
