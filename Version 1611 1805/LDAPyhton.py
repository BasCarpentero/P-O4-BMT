import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDApython

from scipy.fft import fft, fftfreq
from scipy.signal import freqz
from scipy.io import loadmat
import scipy.fftpack
import scipy


import CSP
import Filter
import AuxiliaryFunctions
import LDA
from scipy.io import loadmat
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Load the given Matlab data.
    #data = loadmat('/Users/ogppr/Documents/dataSubject8.mat')
    data = loadmat('dataSubject8.mat')

    # Data detailing which sample attends to which ear.
    wrapped_attended_ear = np.array(data.get('attendedEar'))
    attended_ear = AuxiliaryFunctions.unwrap_cell_data(wrapped_attended_ear)

    # EEG measured data.
    wrapped_EEG_data = np.array(data.get('eegTrials'))
    EEG_data = AuxiliaryFunctions.unwrap_cell_data(wrapped_EEG_data)


    # Calculate a Butterworth bandpass filter for the given parameters.
    fs = int(data.get('fs'))
    low_cut = 12.0
    high_cut = 30.0

    filtered_EEG_data = []
    for minute in EEG_data:
        minute = np.transpose(minute) # rows = 24 channels , columns = 7200 time instances
        y = Filter.butter_bandpass_filter(minute, low_cut, high_cut, fs, order=8)
        y = np.transpose(y)
        y = y[100:]
        filtered_EEG_data.append(y)
    filtered_EEG_data = np.array(filtered_EEG_data)

    plt.figure(4)
    channel = 0
    EEG_data_plot = np.transpose(filtered_EEG_data[0])
    while channel < 24 :
        EEG_data_plot[channel]= np.add(EEG_data_plot[channel], np.full((7100,), channel*(-100)))
        channel += 1
    plt.plot(EEG_data_plot.T, label='Filtered signal')
    plt.xlabel('time (samples per seconds)')
    #plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    #plt.savefig('pythonfilterOrde8')
    plt.show()



    # Case 1: training 1-36, verification 36-48
    print("Case 1: train 1-36, test 36-48")
    attended_ear_1 = np.delete(attended_ear, np.s_[36:48], axis=0)
    EEG_data_1 = np.delete(filtered_EEG_data, np.s_[36:48], axis=0)
    # CSP training
    grouped_data = AuxiliaryFunctions.group_by_class(EEG_data_1, attended_ear_1)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    spatial_dim = 6
    W = CSP.CSP(class_covariances, spatial_dim)
    # LDA training
    trainMinutes = []
    for i in range(0, 36):
        trainMinutes.append(i)
    f = LDA.calculate_f(trainMinutes, W, filtered_EEG_data)
    lda = LDApython()
    lda.fit(f,attended_ear_1)
    #verification
    testMinutes = []
    for i in range(36, 48):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)
    resultaat = lda.predict(f)
    count = 0
    for i in range(12):
        if attended_ear[36+i] != resultaat[i]:
            count += 1
    print((100 - (count * 100 / 12)), "%")


    # Case 2: training 12-48, verficiation 0-12
    print("Case 2: train 12-48, test 0-12")
    attended_ear_2 = np.delete(attended_ear, np.s_[:12], axis=0)
    EEG_data_2 = np.delete(filtered_EEG_data, np.s_[:12], axis=0)
    # CSP training
    grouped_data = AuxiliaryFunctions.group_by_class(EEG_data_2, attended_ear_2)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    spatial_dim = 6
    W = CSP.CSP(class_covariances, spatial_dim)
    # LDA training
    trainMinutes = []
    for i in range(12, 48):
        trainMinutes.append(i)
    f = LDA.calculate_f(trainMinutes, W, filtered_EEG_data)
    lda = LDApython()
    lda.fit(f,attended_ear_2)
    # Verification
    testMinutes = []
    for i in range(0, 12):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)
    resultaat = lda.predict(f)
    count = 0
    for i in range(12):
        if attended_ear[i] != resultaat[i]:
            count += 1
    print((100 - (count * 100 / 12)), "%")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)

    # case 3: training 0-12 + 24-48, verification 12-24
    print("Case 3: train 0-12+24-48, test 12-24")
    attended_ear_3 = np.delete(attended_ear, np.s_[24:36], axis=0)
    EEG_data_3 = np.delete(filtered_EEG_data, np.s_[24:36], axis=0)
    # CSP training
    grouped_data = AuxiliaryFunctions.group_by_class(EEG_data_3, attended_ear_3)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    spatial_dim = 6
    W = CSP.CSP(class_covariances, spatial_dim)
    # LDA training
    trainMinutes = []
    for i in range(0, 12):
        trainMinutes.append(i)
    for i in range(24, 48):
        trainMinutes.append(i)
    f = LDA.calculate_f(trainMinutes, W, filtered_EEG_data)
    lda = LDApython()
    lda.fit(f,attended_ear_3)
    # Verification
    testMinutes = []
    for i in range(12, 24):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)
    resultaat = lda.predict(f)
    count = 0
    for i in range(12):
        if attended_ear[12 + i] != resultaat[i]:
            count += 1
    print((100 - (count * 100 / 12)), "%")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)

    # Case 4: training 1-24 + 36-48, verification 24-36
    print("Case 4: train 1-24+34-48, test 24-36")
    attended_ear_4 = np.delete(attended_ear, np.s_[12:24], axis=0)
    EEG_data_4 = np.delete(filtered_EEG_data, np.s_[12:24], axis=0)
    # CSP training
    grouped_data = AuxiliaryFunctions.group_by_class(EEG_data_4, attended_ear_4)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    spatial_dim = 6
    W = CSP.CSP(class_covariances, spatial_dim)
    # LDA training
    trainMinutes = []
    for i in range(0, 24):
        trainMinutes.append(i)
    for i in range(36, 48):
        trainMinutes.append(i)
    f = LDA.calculate_f(trainMinutes, W, filtered_EEG_data)
    lda = LDApython()
    lda.fit(f,attended_ear_4)
    # verificiation
    testMinutes = []
    for i in range(24, 36):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)
    resultaat = lda.predict(f)
    count = 0
    for i in range(12):
        if attended_ear[24 + i] != resultaat[i]:
            count += 1
    print((100 - (count * 100 / 12)), "%")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)


    # Case 5: training 1-36, verification 24-36
    print("Case 5: train 1-24, test 1-24")
    EEG_data_5 = filtered_EEG_data[:24]
    attended_ear_5 = attended_ear[:24]
    # CSP training
    grouped_data = AuxiliaryFunctions.group_by_class(EEG_data_5, attended_ear_5)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    size = 6
    W = CSP.CSP(class_covariances, size)
    # LDA training
    trainMinutes = []
    for i in range(0, 24):
        trainMinutes.append(i)
    f = LDA.calculate_f(trainMinutes, W, filtered_EEG_data) #dimensies: 36x24
    lda = LDApython(store_covariance=True)
    lda.fit(f, attended_ear_5)
    # Verificiation
    testMinutes = []
    for i in range(0, 24):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)
    resultaat = lda.predict(f)
    count = 0
    for i in range(24):
        if attended_ear[i] != resultaat[i]:
            count += 1
    print((100 - (count * 100 / 24)), "%")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)

