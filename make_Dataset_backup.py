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

#creating one-hot-clips for all data, joining them together, and then random splitting. 
DCPP_all_D = DCPP_all_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=3,class_subset=['D'])
DCPP_all_A = DCPP_all_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['A NE Pacific'])
DCPP_all_B = DCPP_all_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['B NE Pacific'])

# overlap was different for different calls, now I have to join all of the rows together based on their columns
new = DCPP_all_D.join(DCPP_all_A)
DCPP_clips = new.join(DCPP_all_B)

train_clips, validate_clips = sklearn.model_selection.train_test_split(DCPP_clips, train_size=0.7, random_state=0) # use this function to randomly subset them and spit out two new dataframes

# now save each of these as a csv for training! 
validate_clips.to_csv('/home/michaela/CV4E_oss/pre_processing/labeled_data/validate_clips.csv', index=True)
train_clips.to_csv('/home/michaela/CV4E_oss/pre_processing/labeled_data/train_clips.csv', index=True)

# now do the same with test data
CINMS18B = pd.read_csv('/mnt/ssd-cluster/michaela/data/annotations/modified_annotations/CINMS18B_logs_all_MNA_modification.csv')
CINMS18B_box = opensoundscape.BoxedAnnotations(CINMS18B)

CINMS18B_clip_D = CINMS18B_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=3,class_subset=['D'])
CINMS18B_clip_A = CINMS18B_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['A NE Pacific'])
CINMS18B_clip_B = CINMS18B_box.one_hot_clip_labels(clip_duration=15,clip_overlap=0,min_label_overlap=5,class_subset=['B NE Pacific'])

new_clip = CINMS18B_clip_D.join(CINMS18B_clip_A)
CINMS18B_clips = new_clip.join(CINMS18B_clip_B)

CINMS18B_clips.to_csv('/home/michaela/CV4E_oss/pre_processing/labeled_data/CINMS18B_test_clips.csv', index=False)