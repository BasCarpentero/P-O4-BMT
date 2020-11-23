"""
This script contains the main function code.
"""

from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.io import loadmat
import scipy.fftpack
import scipy


import numpy as np

import AuxiliaryFunctions
import Filter
import CSP
import LDA


if __name__ == "__main__":

    # Load the given Matlab data.
    #data = loadmat('/Users/ogppr/Documents/dataSubject8.mat')
    data = loadmat('dataSubject8.mat')

    # Data detailing which sample attends to which ear.
    wrapped_attended_ear = np.array(data.get('attendedEar'))
    attended_ear = AuxiliaryFunctions.unwrap_cell_data(wrapped_attended_ear)
    print('attended_ear is the following list of length ' +
          str(len(attended_ear)) + ' : ' + str(attended_ear) + '.')
    # plt.figure(1)
    # plt.plot(attended_ear, 'rx')
    # plt.xlabel('Sample number')
    # plt.ylabel('Group')

    # EEG measured data.
    wrapped_EEG_data = np.array(data.get('eegTrials'))
    EEG_data = AuxiliaryFunctions.unwrap_cell_data(wrapped_EEG_data)
    print('EEG_data is a list of length ' + str(len(EEG_data)) +
          ' containing 24 channels and 7200 data points per minute.')
    # plt.figure(2)
    # plt.plot(EEG_data[0])
    # plt.xlabel('Time samples minute one')
    # plt.ylabel('Voltage')
    # plt.show()


    # Calculate a Butterworth bandpass filter for the given parameters.
    fs = 120.0
    low_cut = 12.0
    high_cut = 30.0
    # plt.figure(3)
    # plt.clf()
    # for order in [3, 6, 9]:
    #      b, a = Filter.butter_bandpass(low_cut, high_cut, fs, order=order)
    #      w, h = freqz(b, a, worN=2000)
        # plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
    #          '--', label='sqrt(0.5)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain')
    # plt.grid(True)
    # plt.legend(loc='best')
    filtered_EEG_data = []
    for minute in EEG_data:
        minute = np.transpose(minute) # rows = 24 channels , columns = 7200 time instances
        y = Filter.butter_bandpass_filter(minute, low_cut, high_cut, fs, order=8)
        y = np.transpose(y)
        y = y[100:]
        filtered_EEG_data.append(y)


    # plt.figure(4)
    # channel = 0
    # EEG_data_plot = np.transpose(filtered_EEG_data[0])
    # while channel < 24 :
    #     EEG_data_plot[channel]= np.add(EEG_data_plot[channel], np.full((7100,), channel*(-100)))
    #     channel += 1
    # plt.plot(EEG_data_plot.T, label='Filtered signal')
    # plt.xlabel('time (samples per seconds)')
    # #plt.hlines([-a, a], 0, T, linestyles='--')
    # plt.grid(True)
    # plt.axis('tight')
    # #plt.savefig('pythonfilterOrde8')
    # plt.show()


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
    cov_mat = AuxiliaryFunctions.covariance_matrix(np.transpose(f))
    inv_cov_mat = np.linalg.inv(cov_mat)
    f_in_classes = AuxiliaryFunctions.group_by_class(f, attended_ear_1)
    print(np.shape(f_in_classes))
    mean1 = LDA.calculate_mean(np.array(f_in_classes[0]))
    mean2 = LDA.calculate_mean(np.array(f_in_classes[1]))
    v_t, b = LDA.calculate_vt_b(inv_cov_mat, mean1, mean2)

    print(attended_ear_1)

    ###plots###
    for i in range(np.shape(f_in_classes)[1]):
        plt.scatter(f_in_classes[0][i][0], f_in_classes[0][i][5], color='green')
        plt.scatter(f_in_classes[1][i][0],f_in_classes[1][i][5],color='red')
    plt.legend(("Class 1", "Class 2"))
    plt.title("Feature vectors 2D")
    # plt.xlabel('f')
    # plt.ylabel('f')
    plt.show()
    plt.close()


    # v = np.transpose(v_t)
    # x = np.linspace(1, 6, 6)
    # plt.figure("Plot van f(12 x 6)")
    # for i in range(0,12):
    #     plt.scatter(x,f[i])
    # plt.figure("Plot van v(6x1)")
    # plt.scatter(x,v)
    # #plt.plot([1,6], [0,0], 'r--')
    # plt.figure("Plot van v_t * f")
    # y = v_t * f
    # for i in range(0,12):
    #     plt.scatter(x,y[i])
    # #plt.plot([1,6], [0,0], 'r--')




    # Verificication
    testMinutes = []
    for i in range(36, 48):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)

    plt.figure("Classification")
    xas = []
    classification = []
    D = []
    count = 0
    for i in range(12):
        D.append(np.dot(v_t, f[i]) + b)
        xas.append(i+1)
        classification.append(LDA.classify(v_t, b, f[i]))
        if attended_ear[36+i] != LDA.classify(v_t, b, f[i]):
            count += 1
    print((100 - (count * 100 / 12)), "%")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)
    plt.scatter(xas,classification)
    plt.figure("D(f)")
    plt.scatter(xas,D)
    plt.plot([1, 12], [0, 0], 'r--')
    plt.show()
    plt.close()


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
    cov_mat = AuxiliaryFunctions.covariance_matrix(np.transpose(f))
    inv_cov_mat = np.linalg.inv(cov_mat)
    f_in_classes = AuxiliaryFunctions.group_by_class(f, attended_ear_2)
    mean1 = LDA.calculate_mean(np.array(f_in_classes[0]))
    mean2 = LDA.calculate_mean(np.array(f_in_classes[1]))
    v_t, b = LDA.calculate_vt_b(inv_cov_mat, mean1, mean2)
    # Verification
    testMinutes = []
    for i in range(0, 12):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)

    count = 0
    plt.figure("Classification")
    xas = []
    classification = []
    D = []
    print(attended_ear_2)
    for i in range(12):
        D.append(np.dot(v_t, f[i]) + b)
        xas.append(i+1)
        classification.append(LDA.classify(v_t, b, f[i]))
        if attended_ear[i] != LDA.classify(v_t, b, f[i]):
            count += 1
    print((100 - (count*100/12)), "%")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)
    plt.scatter(xas,classification)
    plt.figure("D(f)")
    plt.scatter(xas,D)
    plt.plot([1, 12], [0, 0], 'r--')
    plt.show()
    plt.close()


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
    cov_mat = AuxiliaryFunctions.covariance_matrix(np.transpose(f))
    inv_cov_mat = np.linalg.inv(cov_mat)
    f_in_classes = AuxiliaryFunctions.group_by_class(f, attended_ear_3)
    mean1 = LDA.calculate_mean(np.array(f_in_classes[0]))
    mean2 = LDA.calculate_mean(np.array(f_in_classes[1]))
    v_t, b = LDA.calculate_vt_b(inv_cov_mat, mean1, mean2)
    # Verification
    testMinutes = []
    for i in range(12, 24):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)
    count = 0
    for i in range(12):
        if attended_ear[12+i] != LDA.classify(v_t, b, f[i]):
            count += 1
    print((100 - (count * 100 / 12)), "%")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)

    # Case 4: training 1-24 + 36-48, verification 24-36
    print("Case 4: train 1-24+34-48, test 24-36")
    attended_ear_4 = np.delete(attended_ear, np.s_[12:24], axis=0)
    EEG_data_4 = np.delete(filtered_EEG_data, np.s_[12:24], axis=0)
    #CSP training
    grouped_data = AuxiliaryFunctions.group_by_class(EEG_data_4, attended_ear_4)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    spatial_dim = 6
    W = CSP.CSP(class_covariances, spatial_dim)
    #LDA training
    trainMinutes = []
    for i in range(0, 24):
        trainMinutes.append(i)
    for i in range(36, 48):
        trainMinutes.append(i)
    f = LDA.calculate_f(trainMinutes, W, filtered_EEG_data)
    cov_mat = AuxiliaryFunctions.covariance_matrix(np.transpose(f))
    inv_cov_mat = np.linalg.inv(cov_mat)
    f_in_classes = AuxiliaryFunctions.group_by_class(f, attended_ear_4)
    mean1 = LDA.calculate_mean(np.array(f_in_classes[0]))
    mean2 = LDA.calculate_mean(np.array(f_in_classes[1]))
    v_t, b = LDA.calculate_vt_b(inv_cov_mat, mean1, mean2)
    #verificiation
    testMinutes = []
    for i in range(24, 36):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)
    count = 0
    for i in range(12):
        if attended_ear[24+i] != LDA.classify(v_t, b, f[i]):
            count += 1
    print((100 - (count * 100 / 12)), "%")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)

    # Case 5: training 1-36, verification 24-36
    print("Case 5: train 1-24, test 1-24")
    attended_ear_5 = np.delete(attended_ear, np.s_[24:48], axis=0)
    EEG_data_5 = np.delete(filtered_EEG_data, np.s_[24:48], axis=0)
    #CSP training
    grouped_data = AuxiliaryFunctions.group_by_class(EEG_data_5, attended_ear_5)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    spatial_dim = 6
    W = CSP.CSP(class_covariances, spatial_dim)
    #LDA training
    trainMinutes = []
    for i in range(0, 24):
        trainMinutes.append(i)
    f = LDA.calculate_f(trainMinutes, W, filtered_EEG_data)
    cov_mat = AuxiliaryFunctions.covariance_matrix(np.transpose(f))
    inv_cov_mat = np.linalg.inv(cov_mat)
    f_in_classes = AuxiliaryFunctions.group_by_class(f, attended_ear_5)
    mean1 = LDA.calculate_mean(np.array(f_in_classes[0]))
    mean2 = LDA.calculate_mean(np.array(f_in_classes[1]))
    v_t, b = LDA.calculate_vt_b(inv_cov_mat, mean1, mean2)
    #verificiation
    testMinutes = []
    for i in range(0, 24):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, filtered_EEG_data)
    count = 0
    for i in range(24):
        if attended_ear[i] != LDA.classify(v_t, b, f[i]):
            count += 1
    print((100 - (count * 100 / 24)), "%")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)
