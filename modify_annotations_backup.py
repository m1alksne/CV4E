
# script to load triton log annotations and modify them to fit opensoundscapes format
# needs filepath, call label, annotation start and end time
from datetime import datetime
import os
import glob
import opensoundscape
from AudioStreamDescriptor import XWAVhdr
from opensoundscape import Audio, Spectrogram
import random
import pandas as pd
import numpy as np

directory_path = "/mnt/ssd-cluster/michaela/data/annotations/"
all_files = glob.glob(os.path.join(directory_path,'*.xls'))

# function to extract xwav start time and save it
def extract_xwav_start(path):
    xwav_hdr = XWAVhdr(path)
    xwav_start_time = xwav_hdr.dtimeStart
    return xwav_start_time

# function to get annotation start and end time in seconds since start of xwav
# also replaces old file path with new one
# removes fin whale calls
# uses extract_xwav_start to get get file start time for each row

new_path="/mnt/ssd-cluster/michaela/data/xwavs/"
# calculate start and end time of annotation in seconds since start of xwav
def calculate_annotation_seconds(df):
    df['audio_file'] = [in_file.replace("\\","/").replace("E:/SocalLFDevelopmentData/",new_path) for in_file in df['Input file']] # list comprehension for swapping out file path
    df['file_datetime'] = df['audio_file'].apply(extract_xwav_start) # use apply function to apply extract_xwav_datetime to all rows
    df['start_time'] = (df['Start time'] - df['file_datetime']).dt.total_seconds() # convert start time difference to total seconds
    df['end_time'] = (df['End time'] - df['file_datetime']).dt.total_seconds() # convert end time difference to total seconds
    bp_indices = df.index[df['Species Code'] == 'Bp'].tolist() # indices of fin whale calls
    df.drop(bp_indices, inplace=True)  #remove fin whale calls
    noise_indices = df.index[df['Species Code'] == 'Na'].tolist() # indices of noise
    df.drop(noise_indices, inplace=True)  #remove noise annotations 
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
new_path="/mnt/ssd-cluster/michaela/data/xwavs/"

for file in all_files:
    data = pd.read_excel(file)
    subset_df = calculate_annotation_seconds(data)
    filename = os.path.basename(file)
    new_filename = filename.replace('.xls', '_modification.csv')
     # Construct the path to save the modified DataFrame as a CSV file
    save_path = os.path.join(subfolder_path, new_filename)
    # Save the subset DataFrame to the subset folder as a CSV file
    subset_df.to_csv(save_path, index=False)