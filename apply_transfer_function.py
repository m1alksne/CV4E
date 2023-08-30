# apply transfer function to oss spectrogram object
# written by Sam Lapp to apply hydrophone calibration funcion to spectrogram object

import numpy as np
import opensoundscape
from pathlib import Path 
from opensoundscape.preprocess.actions import BaseAction
import pandas as pd


class TransferFunction(BaseAction):
    """Apply HARP Transfer function to spectrogram object 
        Transfer function will be interpolated and added to spectrogram object
        specify decibel limits for normalizing spectrogram. Indexes into xwav file path and extracts name of xwav, and applies the proper transfer fuction
        based on the tf_key. 
        modify tf_key when running certain datasets

    """
    def __init__(self, decibel_limits=None):
        super(TransferFunction,self).__init__()
        self.params['decibel_limits']=decibel_limits 


        tf_dir = '/mnt/ssd-cluster/michaela/data/transfer_functions/' # path to transfer function
        tf_dict = {'DCPP01A': '688_130415_B_HARP_DCPP01A.tf', # transfer function keys 
                    'DCPP02A': '686_121005_B_HARP_DCPP02A.tf',
                    'CINMS': '618_101105_B_HARP_CINMS18B.tf'}
        tf_dataframe_dict = {}

        for file in tf_dict.keys():

            tf_filename = tf_dict[file]
            tf_path = Path(tf_dir) / tf_filename
            TF = pd.read_csv(tf_path,delim_whitespace=True,header=None)
            TF.columns=['frequency','calibration']
            tf_dataframe_dict[file] = TF
        
        self.tf_dataframe_dict=tf_dataframe_dict 

    def go(self,sample,**kwargs):
        path = Path(sample.source)
        deployment_name = path.name.split('_')[0]

        tf_dataframe = self.tf_dataframe_dict[deployment_name]

        sample.data = apply_transfer_function(sample.data, tf_dataframe, self.params['decibel_limits'])
    

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