
# written by Sam Lapp & MNA to apply bit depth transform to convert OSS wavform to SWAL waveform
import numpy as np
# takes an xwav (that is now an oss object) and converts it to bit size format
def convert_audio_to_bits(xwav):

    bits = 16 
    abs_max = 2 ** (bits - 1)
    xwav.samples = np.float64(xwav.samples)*abs_max

    return xwav