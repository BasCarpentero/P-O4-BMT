"""
This script contains the code for feature vector calculation and linear discriminant analysis.
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

# Returns the log-energy vector computed with T samples given the CSP-filtered data y.
def energy(y, decisionWindow):
    outputEnergyVector = np.zeros(len(y))
    for j in range(decisionWindow):
        for i in range(0, len(y)):
            outputEnergyVector[i] += (y[i][j])**2
    return np.log(outputEnergyVector)


def calculate_f(testMinutes, W, x, T=7190):
    f = []
    for i in testMinutes:
        # x is a C x A matrix with C = #channels and A = #time instances
        # W is a C x K matrix with C = #channels and K = # spatial filters
        # T is the decisionwindow
        xi = np.transpose(x[i][0])
        y = np.dot(np.transpose(W), xi)
        f.append(energy(y, T))
    return np.array(f)


# class 1: LEFT
# class 2: RIGHT
def group_by_class(f, classes):
    class_one = []
    class_two = []
    for x in classes:
        if x == 1:
            class_one.append(np.transpose(f[x]))
        else:
            class_two.append(np.transpose(f[x]))
    return np.array([class_one, class_two], dtype=object)


def calculate_covariance_matrix(data):
    x = np.vstack(data.transpose())
    covMatrix = np.cov(x)
    return covMatrix


def calculate_mean(data):
    data = np.transpose(data)
    means = []
    for i in range(np.shape(data)[0]):
        means.append(np.mean(data[i]))
    return means



def calculate_vt_b(inv_cov_mat, m1, m2):
    diff_mean = np.subtract(m2, m1)
    sum_mean = np.add(m1,m2)
    v = np.dot(inv_cov_mat, diff_mean)
    v_t = v.transpose()
    b = -0.5 * np.dot(v_t, sum_mean)
    return v_t, b


def classify(v_t, b, f):
    if np.dot(v_t, f) + b > 0:
        return 1
    else:
        return 2

if __name__ == "__main__":
    data = loadmat('dataSubject8.mat')

    # Data detailing which sample attends to which ear.
    wrapped_attended_ear = np.array(data.get('attendedEar'))
    attended_ear = AuxiliaryFunctions.unwrap_cell_data(wrapped_attended_ear)


    wrapped_EEG_data = np.array(data.get('eegTrials'))
    EEG_data = AuxiliaryFunctions.unwrap_cell_data(wrapped_EEG_data)

    wrapped_x = np.array(data.get('eegTrials'))


    print("Case 5: train 1-24, test 1-24")
    attended_ear_5 = np.delete(attended_ear, np.s_[24 - 48], axis=0)
    EEG_data_5 = np.delete(EEG_data, np.s_[24 - 48], axis=0)
    # CSP training
    grouped_data = CSP.group_by_class(EEG_data_5, attended_ear_5)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    W = CSP.CSP(class_covariances)
    # LDA training
    trainMinutes = []
    for i in range(0, 24):
        trainMinutes.append(i)
    f = LDA.calculate_f(trainMinutes, W, wrapped_x)
    cov_mat = LDA.calculate_covariance_matrix(f)
    inv_cov_mat = np.linalg.inv(cov_mat)
    f_in_classes = LDA.group_by_class(f, attended_ear_5)
    mean1 = LDA.calculate_mean(np.array(f_in_classes[0]))
    mean2 = LDA.calculate_mean(np.array(f_in_classes[1]))
    v_t, b = LDA.calculate_vt_b(inv_cov_mat, mean1, mean2)
    # verificiation
    testMinutes = []
    for i in range(0, 24):
        testMinutes.append(i)
    f = LDA.calculate_f(testMinutes, W, wrapped_x)
    count = 0
    for i in range(24):
        if attended_ear[i] != LDA.classify(v_t, b, f[i]):
            count += 1
    print((100 - (count * 100 / 24)), "%")
