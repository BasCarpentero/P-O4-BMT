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
# class 1: LEFT
# class 2: RIGHT

# Returns the log-energy vector computed with T samples given the CSP-filtered data y.
def logenergy(y):
    outputEnergyVector = np.zeros(len(y))
    for i in range(len(y)):
        outputEnergyVector[i] = sum(j**2 for j in y[i])
    return np.log(outputEnergyVector)


def calculate_f(testMinutes, W, x):
    f = []
    for i in testMinutes:
        # x is a C x A matrix with C = #channels and A = #time instances
        # W is a C x K matrix with C = #channels and K = # spatial filters
        # T is the decisionwindow
        y = np.dot(np.transpose(W), np.transpose(x[i]))
        f.append(logenergy(y))
    return np.array(f)


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
    if np.dot(v_t, f) + b < 0:
        return 1
    else:
        return 2
