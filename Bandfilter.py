# This script contains the code for a filterbank in the first step of directional EEG decoding.
# This library contains the necessary tools to construct passband filters.
from scipy.signal import butter, lfilter


# Calculates an IIR filter with given lowcut and highcut cutting frequencies and the desired order.
def butter_bandpass(lowcut, highcut, sampling_frequency, order=5):
    nyquist_frequency = 0.5 * sampling_frequency
    low = lowcut / nyquist_frequency
    high = highcut / nyquist_frequency
    b, a = butter(order, [low, high], btype='band')  # IIR filter constants
    return b, a


# Applies a bandpass filter to the given data.
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

'''
def filtersignaal(startMinute,endMinute, inputSignal, lowcut, highcut, fs, order=8):
    y = zeros(endMinute-startMinute)
    for t in range(startMinute,endMinute+1):
        x_tT = inputSignal[t][0].T
        y[t] = butter_bandpass_filter(x_tcT, lowcut, highcut, fs, order)
    return y
'''



# Example; should work for the data that will be given to us.
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz
    import scipy.io
    data = scipy.io.loadmat('dataSubject8.mat')
    fs = float(data.get('fs'))
    #betaband:
    lowcut = 12.0
    highcut = 30.0

    '''
    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    '''

    # EEG data given for this project.
    #a = 0.02
    x = np.array(data.get('eegTrials'))
    #eerste minuut:
    x_00T = x[0][0].transpose()


    y = butter_bandpass_filter(x_00T, lowcut, highcut, fs, order=8)

    #REMOVING TRANSIENT PART
    y = y[:,100:]

    #PLOTTING FILTERED SIGNAL
    plt.figure()
    plt.plot(y.T, label='Filtered signal')
    plt.xlabel('time (samples per seconds)')
    #plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.savefig('pythonfilterOrde8')
    #plt.legend(loc='upper right')

    plt.show()
