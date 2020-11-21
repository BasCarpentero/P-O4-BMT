"""
This script contains the code for a filterbank in the first step of directional EEG decoding.
"""

from scipy.signal import butter, lfilter


# Calculates an IIR filter with given lower cut and higher cut cutting frequencies and the desired order.
def butter_bandpass(low_cut, high_cut, sampling_frequency, order):
    nyquist_frequency = 0.5 * sampling_frequency
    low = low_cut / nyquist_frequency
    high = high_cut / nyquist_frequency
    b, a = butter(order, [low, high], btype='band')  # IIR filter constants
    return b, a


# Applies a bandpass filter to given data.
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
