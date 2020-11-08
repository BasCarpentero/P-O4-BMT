import numpy as np
from scipy.io import loadmat
import CSP


def createFeature(y, T):
    #@param: x is a C x A matrix with C = #channels and A = #time instances
    #@param: W is a C x K matrix with C = #channels and K = # spatial filters
    #@param T: number of time samples in the decision window
    outputEnergy = np.zeros(len(y))
    for j in range(T):
        for i in range(0, len(y)):
            outputEnergy[i] += (y[i][j])**2
    f = np.log(outputEnergy)
    return f


if __name__ == "__main__":
    data = loadmat('dataSubject8.mat')
    wrapped_attended_ear = np.array(data.get('attendedEar'))[:36]
    attended_ear = CSP.unwrap_cell_data(wrapped_attended_ear)
    wrapped_EEG_data = np.array(data.get('eegTrials'))[:36]
    EEG_data = CSP.unwrap_cell_data(wrapped_EEG_data)

    grouped_data = CSP.group_by_class(EEG_data, attended_ear)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    W = CSP.CSP(class_covariances)

    wrapped_x = np.array(data.get('eegTrials'))
    x = wrapped_x[36][0].T
    y = np.dot(W.T, x)

    T = 7170
    f = createFeature(y, T)
    print(f)
