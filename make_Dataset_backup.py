# script to make opensoundscape datasets for training, validation, and test!
# will be modified later once I have more data

import opensoundscape
import glob
import os
import pandas as pd
import numpy as np
import sklearn
import librosa
import torch
import random

# read in the files that you want 

# smoosh all of DCPP data together for train and validate datasets
DCPP1 = pd.read_csv('/mnt/ssd-cluster/michaela/data/annotations/modified_annotations/DCPP02A_logs_summer_modification.csv')
DCPP2 = pd.read_csv('/mnt/ssd-cluster/michaela/data/annotations/modified_annotations/DCPP01A_logs_all_MNA_modification.csv')
DCPP3 = pd.read_csv('/mnt/ssd-cluster/michaela/data/annotations/modified_annotations/DCPP02A_logs_spring_modification.csv')
DCPP_all = pd.concat([DCPP1,DCPP2,DCPP3],ignore_index=True)
DCPP_all_box = opensoundscape.BoxedAnnotations(DCPP_all)
DCPP_all_box = opensoundscape.BoxedAnnotations(DCPP_all)
DCPP_all_box.audio_files =  DCPP_all['audio_file'].unique()

#creating one-hot-clips for all data, joining them together, and then random splitting. 
DCPP_all_D = DCPP_all_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=3,class_subset=['D'])
DCPP_all_A = DCPP_all_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['A NE Pacific'])
DCPP_all_B = DCPP_all_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['B NE Pacific'])

# overlap was different for different calls, now I have to join all of the rows together based on their columns
new = DCPP_all_D.join(DCPP_all_A)
DCPP_clips = new.join(DCPP_all_B)
DCPP_clips.to_csv('/home/michaela/CV4E/labeled_data/DCPP_one_hot_clips.csv', index=False)

train_clips, validate_clips = sklearn.model_selection.train_test_split(DCPP_clips, train_size=0.7, random_state=0) # use this function to randomly subset them and spit out two new dataframes
path_to_remove = '/mnt/ssd-cluster/michaela/data/xwavs/DCCP01A_fall/DCPP01A_d01_121115_054102.d100.x.wav'
train_clips = train_clips.reset_index()
train_clips_new = train_clips[train_clips['file'] != path_to_remove] # this will need to be modified for column indices
balanced_train_clips = opensoundscape.data_selection.resample(train_clips,n_samples_per_class=1500,random_state=0) # upsample (repeat samples) so that all classes have 1000 samples
balanced_train_clips_standard = balanced_train_clips.reset_index()


# must
train_clips_new = train_clips_new.reset_index(drop=True)
train_clips_new
filtered_indices = train_clips_new.index[(train_clips_new['D'] == 0) & (train_clips_new['A NE Pacific'] == 0) & (train_clips_new['B NE Pacific'] == 0)]# indices of negatives

random_sample_indices = random.sample(filtered_indices.tolist(), 1500)

combined_indices = list(random_sample_indices) 

# and now reapply my filtered indices to the dataframe 
#train_clips_filtered = train_clips[combined_indices]
train_clips_noise = train_clips_new.iloc[random_sample_indices]

train_clips_final = pd.concat([train_clips_noise, balanced_train_clips_standard]).reset_index(drop=True)

train_clips_final.sum()

validate_clips = validate_clips.reset_index()
validate_clips_new = validate_clips[validate_clips['file'] != path_to_remove]
validate_clips_final = validate_clips_new.reset_index(drop=True)
validate_random_samples = validate_clips_final.sample(n=2000, random_state=0)

# now save each of these as a csv for training! 
validate_random_samples.to_csv('/home/michaela/CV4E/labeled_data/validate_clips.csv', index=False)
train_clips_final.to_csv('/home/michaela/CV4E/labeled_data/train_clips.csv', index=False)

# now do the same with test data
CINMS18B = pd.read_csv('/mnt/ssd-cluster/michaela/data/annotations/modified_annotations/CINMS18B_logs_all_MNA_modification.csv')
CINMS18B_box = opensoundscape.BoxedAnnotations(CINMS18B)

CINMS18B_clip_D = CINMS18B_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=3,class_subset=['D'])
CINMS18B_clip_A = CINMS18B_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['A NE Pacific'])
CINMS18B_clip_B = CINMS18B_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['B NE Pacific'])

new_clip = CINMS18B_clip_D.join(CINMS18B_clip_A)
CINMS18B_clips = new_clip.join(CINMS18B_clip_B)

CINMS18B_clips.to_csv('/home/michaela/CV4E/labeled_data/CINMS18B_one_hot_clips.csv', index=False)
