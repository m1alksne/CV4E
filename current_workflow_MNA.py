# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:51:53 2023

@author: MNA 
 modifying workflow to test on new datasets
 1. read in log files from wherever they are saved
 2. convert start and end times to number of seconds since the start of the xwav file. Need MRoch code to read xwav headers
 3. save that as a new "modified_log"
 4. read in modified log and make one_hot_clips
 5. read in xwav from one_hot_clips and plot some spectrograms to make sure they look ok
 6. predict
"""

from datetime import datetime
import matplotlib.pyplot as plt
import os
import glob
import opensoundscape
import sys
sys.path.append(r"C:\Users\DAM1\CV4E")
from AudioStreamDescriptor import XWAVhdr
from opensoundscape import Audio, Spectrogram
import random
import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 1.
directory_path = "L:\CV4E\Other logs\other_old"
all_files = glob.glob(os.path.join(directory_path,'*SOCAL34M_LF_logs_MNA_edits.xls'))

# function to extract xwav start time and save it
def extract_xwav_start(path):
    xwav_hdr = XWAVhdr(path)
    xwav_start_time = xwav_hdr.dtimeStart
    return xwav_start_time

def calculate_annotation_seconds(df):
    df['audio_file'] = df['Input file']
    #df['audio_file'] = [in_file.replace("\\","/").replace("SOCAL34M",new_path) for in_file in df['Input file']] # list comprehension for swapping out file path
    df['file_datetime'] = df['audio_file'].apply(extract_xwav_start) # use apply function to apply extract_xwav_datetime to all rows
    df['start_time'] = (df['Start time'] - df['file_datetime']).dt.total_seconds() # convert start time difference to total seconds
    df['end_time'] = (df['End time'] - df['file_datetime']).dt.total_seconds() # convert end time difference to total seconds
    #bp_indices = df.index[df['Species Code'] == 'Bp'].tolist() # indices of fin whale calls
   # df.drop(bp_indices, inplace=True)  #remove fin whale calls
   # noise_indices = df.index[df['Species Code'] == 'Na'].tolist() # indices of noise
   # df.drop(noise_indices, inplace=True)  #remove noise annotations 
    df['annotation']= df['Call']
    df['high_f'] = df['Parameter 1']
    df['low_f'] = df['Parameter 2']

    #df['Input file'] = [in_file.replace("\\","/").replace("E:/SocalLFDevelopmentData/",new_path) for in_file in df['Input file']] # list comprehension for swapping out file path
    df = df.loc[:, ['audio_file','annotation','high_f','low_f','start_time','end_time']] # subset all rows by certain column name
    return df

# make a subfolder for saving modified logs 
subfolder_name = "modified_annotations"
# Create the subfolder if it doesn't exist
subfolder_path = os.path.join(directory_path, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)

# loop through all annotation files and save them in subfolder "modified_annotations"
#new_path="/mnt/ssd-cluster/michaela/data/xwavs/"
new_path = "L:/CV4E/xwavs/SOCAL34M/"

#2 and 3.
for file in all_files:
    data = pd.read_excel(file)
    subset_df = calculate_annotation_seconds(data)
    filename = os.path.basename(file)
    new_filename = filename.replace('.xls', '_modification.csv')
     # Construct the path to save the modified DataFrame as a CSV file
    save_path = os.path.join(subfolder_path, new_filename)
    # Save the subset DataFrame to the subset folder as a CSV file
    subset_df.to_csv(save_path, index=False)

# 4. 
SOCAL34M = pd.read_csv('L:\CV4E\Other logs\other_old\modified_annotations\SOCAL34M_LF_logs_MNA_edits_modification.csv', index_col=False)
SOCAL34M_all = pd.concat([SOCAL34M], ignore_index=True)
SOCAL34M_all_box =  opensoundscape.BoxedAnnotations(SOCAL34M)
SOCAL34M_all_D = SOCAL34M_all_box.one_hot_clip_labels(clip_duration=15, clip_overlap=0, min_label_overlap=3, class_subset =['D'])
SOCAL34M_all_A = SOCAL34M_all_box.one_hot_clip_labels(clip_duration=15, clip_overlap=0, min_label_overlap=5, class_subset =['A NE Pacific'])
SOCAL34M_all_B = SOCAL34M_all_box.one_hot_clip_labels(clip_duration=15, clip_overlap=0, min_label_overlap=5, class_subset =['B NE Pacific'])
new_clip = SOCAL34M_all_D.join(SOCAL34M_all_A)
SOCAL34M_clips = new_clip.join(SOCAL34M_all_B)
SOCAL34M_clips.to_csv('C:\\Users\DAM1\CV4E\labeled_data\SOCAL34M_one_hot_clips.csv')

SOCAL44N = pd.read_csv('L:\\CV4E\Other logs\modified_annotations\SOCAL44N_DcallGT_Jul_ACR_modification.csv', index_col=False)
SOCAL44N_all = pd.concat([SOCAL44N], ignore_index=True)
SOCAL44N_all_box =  opensoundscape.BoxedAnnotations(SOCAL44N)
SOCAL44_all_D = SOCAL44N_all_box.one_hot_clip_labels(clip_duration=15, clip_overlap=0, min_label_overlap=3, class_subset =['D'])
SOCAL44_all_A = SOCAL44N_all_box.one_hot_clip_labels(clip_duration=15, clip_overlap=0, min_label_overlap=5, class_subset =['A NE Pacific'])
SOCAL44_all_B = SOCAL44N_all_box.one_hot_clip_labels(clip_duration=15, clip_overlap=0, min_label_overlap=5, class_subset =['B NE Pacific'])
new_clip = SOCAL44_all_D.join(SOCAL44_all_A)
SOCAL44N_clips = new_clip.join(SOCAL44_all_B)
SOCAL44N_clips.to_csv('C:\\Users\DAM1\CV4E\labeled_data\SOCAL44N_one_hot_clips.csv')

# one hot clips for CINMS18B

CINMS18B = pd.read_csv('L:\CV4E\Dolapo logs\FINAL\modified_annotations\CINMS18B_logs_all_MNA_modification.csv', index_col=False)
CINMS18B_all = pd.concat([CINMS18B], ignore_index=True)
CINMS18B_all_box =  opensoundscape.BoxedAnnotations(CINMS18B)
CINMS18B_all_D = CINMS18B_all_box.one_hot_clip_labels(clip_duration=15, clip_overlap=0, min_label_overlap=3, class_subset =['D'])
CINMS18B_all_A = CINMS18B_all_box.one_hot_clip_labels(clip_duration=15, clip_overlap=0, min_label_overlap=5, class_subset =['A NE Pacific'])
CINMS18B_all_B = CINMS18B_all_box.one_hot_clip_labels(clip_duration=15, clip_overlap=0, min_label_overlap=5, class_subset =['B NE Pacific'])
new_clip = CINMS18B_all_D.join(CINMS18B_all_A)
CINMS18B_clips = new_clip.join(CINMS18B_all_B)
CINMS18B_clips.to_csv('L:\CV4E\BigBlueWave\CV4E\code\BigBlueWhale-oss\labeled_data\CINMS18B_one_hot_clips_new.csv')


# 5. 
# trying to plot a spectrogram and make sure it works (11/14/23) 

tf_path = 'L:\\CV4E\\transfer function\\588_091116_A_HARP_SOCAL44N.tf'
TF = pd.read_csv(tf_path,delim_whitespace=True,header=None)
TF.columns=['frequency','calibration']
def apply_transfer_function(spec,tf_dataframe,decibel_limits=None):
    """
    apply transfer function to opensoundscape.Spectrogram object
    
    helper function to apply transfer function to Spectrogram
    transfer function is list of | freq | dB offset |
    we should interpolate to the frequencies contained in the specrogram

    Args:
        spec: a Specrogram object
        tf_dataframe: dataframe with columns 'freq' (frequencies in Hz) and 'intensity' (dB offset)
        decibel_limits: default None will use original spectrogram's .decibel_units attribute;
            optionally specify a new decibel_limits range for the returned Spectrogram
    """
    if decibel_limits is None:
        decibel_limits = spec.decibel_limits
        
    #extract frequency column and intensity column from transfer function dataframe
    transfer_function_freqs = tf_dataframe.frequency.values
    transfer_function_offsets = tf_dataframe.calibration.values
    
    # linearly interpolate the frequencies from the transfer function table
    # onto the frequencies of the spectrogram to get offsets for each spectrogram row
    spec_offsets = np.interp(spec.frequencies,transfer_function_freqs, transfer_function_offsets)
    
    # add the offset values to each row of the spectrogram
    new_spec_values = (spec.spectrogram.transpose() + np.array(spec_offsets)).transpose()
    
    #create a new spectrogram object with the new values
    return opensoundscape.Spectrogram(new_spec_values,times=spec.times,frequencies=spec.frequencies,decibel_limits=decibel_limits)
file_path = 'L:\BigBlueWave\old\D Call - 40 Hz rCNN\df100_data\SOCAL44N\SOCAL4$_disk04_110705_155500.df100.x.wav'
D_fp = opensoundscape.Audio.from_file(file_path, sample_rate=2000, offset=0, duration=15)
bits = 16 
abs_max = 2 ** (bits - 1)

# Scale the audio
D_fp.samples = np.float64(D_fp.samples) * abs_max
    
# Create a spectrogram
D_fp1 = opensoundscape.Spectrogram.from_audio(D_fp, window_type='hamming', window_samples=1000, 
                                                  overlap_samples=900, fft_size=2000, 
                                                  decibel_limits=(-200,200), scaling='density')
    
# Apply the transfer function
D_fp1_TF = apply_transfer_function(D_fp1, TF, decibel_limits=(40, 140))
    
# Bandpass filter and plot
   
filtered_image = D_fp1_TF.bandpass(10, 150).to_image()
filtered_image.size

# Display the image using matplotlib
plt.imshow(filtered_image)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()

#6. 

# Correct file path
model_path = 'C:\\Users\\DAM1\\CV4E\\model_states\\best.model'

# Load the model
model = opensoundscape.ml.cnn.load_model(model_path)

# for test data!

model.preprocessor.pipeline.to_spec.params.window_samples = 1000 # 100 window samples
model.preprocessor.pipeline.to_spec.params.overlap_samples = 900 # 90% overlap, for 2000 Fs this means 900 samples, and 0.05 sec bins
model.preprocessor.pipeline.to_spec.params.fft_size = 2000 # FFT = Fs, 1 Hz bins
model.preprocessor.out_shape = [224,448,3] # resize image the size that I want ? might not work with pre-trained weights ?

# load data 
test_clips_SOCALN = pd.read_csv('L:\CV4E\BigBlueWave\CV4E\code\BigBlueWhale-oss\labeled_data\SOCAL44N_one_hot_clips_mini.csv', index_col=[0,1,2])
test_clips_CINMS18B = pd.read_csv('L:\CV4E\BigBlueWave\CV4E\code\BigBlueWhale-oss\labeled_data\CINMS18B_one_hot_clips.csv', index_col=[0,1,2])

test_scores_CINMS18B = model.predict(test_clips_CINMS18B)


test_scores_CINMS18B.columns = ['pred_D','pred_A','pred_B']
test_all = test_clips_CINMS18B.join(test_scores_CINMS18B)
test_all.head()
test_evaluation_CINMS18B = test_all.reset_index()
test_evaluation_CINMS18B.to_csv('L:\CV4E\BigBlueWave\CV4E\code\BigBlueWhale-oss\predictions\CINMS18B_predictions.csv')

#test_all["pred_D"] = expit(test_all["pred_D"])

# check D call preformance

D_eval_index = test_evaluation_CINMS18B.index[test_evaluation_CINMS18B['D']==1]
D_eval = test_evaluation_CINMS18B.loc[D_eval_index]
D_noise_index = test_evaluation_CINMS18B.index[test_evaluation_CINMS18B['D']==0]
D_noise = test_evaluation_CINMS18B.loc[D_noise_index]

plt.hist(D_noise['pred_D'],bins=40,alpha=0.5,edgecolor='black',color='blue',label='Noise prediction score')
plt.hist(D_eval['pred_D'],bins=40,alpha=0.5,edgecolor='black',color='orange',label='D call prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('D call prediction scores test CINMS18B')
plt.legend(loc='upper right')

precision, recall, thresholds = precision_recall_curve(test_evaluation_CINMS18B['D'], test_evaluation_CINMS18B['pred_D'])

#create precision recall curve for blue whale B calls
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')
#add axis labels to plot
ax.set_title('Precision-Recall Curve D calls CINMS18B')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
#display plot
plt.show()

threshold_index = np.argmax(precision >= 0.4) # based on PR curve.....
chosen_threshold = thresholds[threshold_index]
binary_D_predictions = (test_evaluation_CINMS18B['pred_D'] >= chosen_threshold).astype(int) # I think this converts to zeros and ones for the calls above this prediction score...
accuracy_value = accuracy_score(test_evaluation_CINMS18B['D'], binary_D_predictions)
precision_value = precision_score(test_evaluation_CINMS18B['D'], binary_D_predictions)
recall_value = recall_score(test_evaluation_CINMS18B['D'], binary_D_predictions)
f1 = f1_score(test_evaluation_CINMS18B['D'], binary_D_predictions)
conf_matrix = confusion_matrix(test_evaluation_CINMS18B['D'], binary_D_predictions)
print(conf_matrix)


# check B call preformance

# B call test
B_eval_index = test_evaluation.index[test_evaluation['B NE Pacific']==1]
B_eval = test_evaluation.loc[B_eval_index]
B_noise_index = test_evaluation.index[test_evaluation['B NE Pacific']==0] 
B_noise = test_evaluation.loc[B_noise_index]

plt.hist(B_noise['pred_B'],bins=40,alpha=0.5,edgecolor='black',color='blue',label='Noise prediction score')
plt.hist(B_eval['pred_B'],bins=40,alpha=0.5,edgecolor='black',color='orange',label='B call prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('B call prediction scores test')
plt.legend(loc='upper right')

precision, recall, thresholds = precision_recall_curve(test_evaluation['B NE Pacific'], test_evaluation['pred_B'])

#create precision recall curve for blue whale B calls
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')
#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
#display plot
plt.show()
# select threshold

# Find the index where precision is high and recall is also reasonably high
# You can adjust the condition based on your specific requirements
threshold_index = np.argmax(precision >= 0.8)
# Use the threshold at that index
chosen_threshold = thresholds[threshold_index]
binary_B_call_predictions = (test_evaluation['pred_B'] >= chosen_threshold).astype(int)
# so "binary_B_call_preditctions" is in theory all of the B calls that we're saying are "true" b/c they are above this threshold


# test SOCALN44
test_scores_SOCAL44N = model.predict(test_clips_SOCALN, num_workers = 12, batch_size = 128)

test_scores_SOCAL44N.columns = ['pred_D','pred_A','pred_B']
test_all_SOCAL44N = test_clips_SOCALN.join(test_scores_SOCAL44N)
test_evaluation_SOCAL44N = test_all_SOCAL44N.reset_index()
test_evaluation_SOCAL44N.to_csv('L:\CV4E\BigBlueWave\CV4E\code\BigBlueWhale-oss\predictions\SOCAL44N_predictions.csv')
#test_all["pred_D"] = expit(test_all["pred_D"])

# check D call preformance

D_eval_index = test_evaluation_SOCAL44N.index[test_evaluation_SOCAL44N['D']==1]
D_eval = test_evaluation_SOCAL44N.loc[D_eval_index]
D_noise_index = test_evaluation_SOCAL44N.index[test_evaluation_SOCAL44N['D']==0]
D_noise = test_evaluation_SOCAL44N.loc[D_noise_index]

plt.hist(D_noise['pred_D'],bins=40,alpha=0.5,edgecolor='black',color='blue',label='Noise prediction score')
plt.hist(D_eval['pred_D'],bins=40,alpha=0.5,edgecolor='black',color='orange',label='D call prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('D call prediction scores test SOCAL44N')
plt.legend(loc='upper right')

precision, recall, thresholds = precision_recall_curve(test_evaluation_SOCAL44N['D'], test_evaluation_SOCAL44N['pred_D'])

#create precision recall curve for blue whale B calls
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')
#add axis labels to plot
ax.set_title('Precision-Recall Curve D calls SOCAL44N')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
#display plot
plt.show()

threshold_index = np.argmax(precision >= 0.4) # based on PR curve.....
chosen_threshold = thresholds[threshold_index]
binary_D_predictions = (test_evaluation['pred_D'] >= chosen_threshold).astype(int) # I think this converts to zeros and ones for the calls above this prediction score...
accuracy_value = accuracy_score(test_evaluation['D'], binary_D_predictions)
precision_value = precision_score(test_evaluation['D'], binary_D_predictions)
recall_value = recall_score(test_evaluation['D'], binary_D_predictions)
f1 = f1_score(test_evaluation['D'], binary_D_predictions)
conf_matrix = confusion_matrix(test_evaluation['D'], binary_D_predictions)
print(conf_matrix)

test_clips_SOCAL34M = pd.read_csv('L:\CV4E\BigBlueWave\CV4E\code\BigBlueWhale-oss\labeled_data\SOCAL34M_one_hot_clips.csv')
# SADLY i MUST coherse my model into thinking this data is from CINMS18B lol so I can use the transfer function
test_clips_SOCAL34M['file'] = test_clips_SOCAL34M['file'].str.replace("SOCAL34M_", "CINMS18B_")
test_clips_SOCAL34M.to_csv('L:\CV4E\BigBlueWave\CV4E\code\BigBlueWhale-oss\labeled_data\SOCAL34M_one_hot_clips_coherced_mini.csv')

test_clips_SOCAL34M = pd.read_csv('L:\CV4E\BigBlueWave\CV4E\code\BigBlueWhale-oss\labeled_data\SOCAL34M_one_hot_clips_coherced.csv',index_col=[0,1,2])



# test SOCAL34m
test_scores_SOCAL34m = model.predict(test_clips_SOCAL34M, num_workers=12, batch_size=128)

test_scores_SOCAL34m.columns = ['pred_D','pred_A','pred_B']
test_all = test_clips_SOCAL34M.join(test_scores_SOCAL34m)
test_all.head()
test_evaluation_SOCAL34M = test_all.reset_index()

test_evaluation_SOCAL34M.to_csv('L:\CV4E\BigBlueWave\CV4E\code\BigBlueWhale-oss\predictions\SOCAL34M_predictions.csv')




