import CSP
import FeatureExtraction
import numpy as np
from scipy.io import loadmat

# class 1: LEFT
# class 2: RIGHT


def get_f(begin, end, W, x):
    f = []
    for i in range(begin, end):
        xi = x[i][0].T
        y = np.dot(W.T, xi)
        T = 7190
        f.append(FeatureExtraction.createFeature(y, T))
    return np.array(f)


def group_by_class(f, classes):
    class_one = []
    class_two = []
    for x in classes:
        if x == 1:
            class_one.append(np.transpose(f[x]))
        else:
            class_two.append(np.transpose(f[x]))
    return np.array([class_one, class_two])


def get_covariance_matrix(data):
    x = np.vstack(data.transpose())
    covMatrix = np.cov(x)
    return np.linalg.inv(covMatrix)


def get_mean(data):
    data = np.transpose(data)
    mean = []
    for i in range(np.shape(data)[0]):
        mean_x = sum(data[i])/len(data[i])
        mean.append(mean_x)
    return mean


def get_vt_b(inv_cov_mat, m1, m2):
    diff_mean = [0] * len(m1)
    sum_mean = [0] * len(m1)
    for i in range(len(m1)):
        diff_mean[i] = m2[i] - m1[i]
        sum_mean[i] = m2[i] + m1[i]
    v = np.dot(inv_cov_mat, diff_mean)
    v_t = v.transpose()
    b = -0.5 * np.dot(v_t, sum_mean)
    return v_t, b


def get_D(v_t, b, f):
    return np.dot(v_t, f) + b


def classify(D):
    if D > 0:
        return 1
    else:
        return 2


if __name__ == "__main__":
    begin, end = 0, 36  # minutes
    data = loadmat('dataSubject8.mat')
    wrapped_attended_ear = np.array(data.get('attendedEar'))
    attended_ear = CSP.unwrap_cell_data(wrapped_attended_ear)[:36]
    wrapped_EEG_data = np.array(data.get('eegTrials'))
    EEG_data = CSP.unwrap_cell_data(wrapped_EEG_data)[begin:end]

    grouped_data = CSP.group_by_class(EEG_data, attended_ear)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    W = CSP.CSP(class_covariances)
    print(W.shape)

    wrapped_x = np.array(data.get('eegTrials'))
    f = get_f(begin, end, W, wrapped_x)
    inv_cov_mat = get_covariance_matrix(f)
    f_in_classes = group_by_class(f, attended_ear)
    mean1 = get_mean(np.array(f_in_classes[0]))
    mean2 = get_mean(np.array(f_in_classes[1]))
    v_t, b = get_vt_b(inv_cov_mat, mean1, mean2)
    attended_ear2 = CSP.unwrap_cell_data(wrapped_attended_ear)[36:]
    f = get_f(36, 48, W, wrapped_x)
    count = 0
    for i in range(12):
        D = get_D(v_t, b, f[i])
        if attended_ear2[i] != classify(D):
            count += 1
    print("Count:", count)  # Aantal verkeerd voorspelde minuten (veel te hoog!!)
