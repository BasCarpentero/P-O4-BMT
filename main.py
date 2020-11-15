import CSP
import numpy as np
from scipy.io import loadmat


# Returns the log-energy vector computed with T samples given the CSP-filtered data y.
def feature(y, T):
    #@param: x is a C x A matrix with C = #channels and A = #time instances
    #@param: W is a C x K matrix with C = #channels and K = # spatial filters
    outputEnergyVector = np.zeros(len(y))
    for i in range(T):
        for j in range(0, len(y)):
            outputEnergyVector[j] += (y[j][i])**2
    return np.log(outputEnergyVector)


def calculate_f(begin, end, W, x):
    f = []
    for i in range(begin, end):
        xi = x[i][0].T
        y = np.dot(W.T, xi)
        T = 7190
        f.append(feature(y, T))
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


def get_covariance_matrix(data):
    x = np.vstack(data.transpose())
    covMatrix = np.cov(x)
    return np.linalg.inv(covMatrix)


def calculate_mean(data):
    data = np.transpose(data)
    mean = []
    for i in range(np.shape(data)[0]):
        mean_x = sum(data[i]) / len(data[i])
        mean.append(mean_x)
    return mean


def calculate_vt_b(inv_cov_mat, m1, m2):
    diff_mean = [0] * len(m1)
    sum_mean = [0] * len(m1)
    for i in range(len(m1)):
        diff_mean[i] = m2[i] - m1[i]
        sum_mean[i] = m2[i] + m1[i]
    v = np.dot(inv_cov_mat, diff_mean)
    v_t = v.transpose()
    b = -0.5 * np.dot(v_t, sum_mean)
    return v_t, b


def calculate_D(v_t, b, f):
    return np.dot(v_t, f) + b


def classify(D):
    if D > 0:
        return 1
    else:
        return 2


if __name__ == "__main__":
    #data = loadmat('/Users/ogppr/Documents/dataSubject8.mat')
    data = loadmat('dataSubject8.mat')
    wrapped_attended_ear = np.array(data.get('attendedEar'))
    attended_ear = CSP.unwrap_cell_data(wrapped_attended_ear)
    attended_ear_1 = CSP.unwrap_cell_data(wrapped_attended_ear)[:36]
    attended_ear_2 = np.delete(attended_ear, np.s_[24-36], axis=0)
    attended_ear_3 = np.delete(attended_ear, np.s_[12-24], axis=0)
    attended_ear_4 = np.delete(attended_ear, np.s_[0-12], axis=0)
    wrapped_EEG_data = np.array(data.get('eegTrials'))
    EEG_data = CSP.unwrap_cell_data(wrapped_EEG_data)
    EEG_data_1 = CSP.unwrap_cell_data(wrapped_EEG_data)[0:36]
    EEG_data_2 = np.delete(EEG_data, np.s_[24-36], axis=0)
    EEG_data_3 = np.delete(EEG_data, np.s_[12-24], axis=0)
    EEG_data_4 = np.delete(EEG_data, np.s_[0-12], axis=0)

    grouped_data = CSP.group_by_class(EEG_data, attended_ear)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    W = CSP.CSP(class_covariances)

    wrapped_x = np.array(data.get('eegTrials'))
    #geval 1: test data 12-48
    begin, end = 12, 48
    f = calculate_f(begin, end, W, wrapped_x)
    inv_cov_mat = get_covariance_matrix(f)
    f_in_classes = group_by_class(f, attended_ear[12:48])
    mean1 = calculate_mean(np.array(f_in_classes[0]))
    mean2 = calculate_mean(np.array(f_in_classes[1]))
    v_t, b = calculate_vt_b(inv_cov_mat, mean1, mean2)
    attended_ear2 = CSP.unwrap_cell_data(wrapped_attended_ear)[:12]
    f = calculate_f(0, 12, W, wrapped_x)
    count = 0
    for i in range(12):
        D = calculate_D(v_t, b, f[i])
        if attended_ear2[i] != classify(D):
            count += 1
    print("Count:", count, (100 - (count*100/12)), "% juist")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)

    # geval 2: verificatie 12-24-> 12-23
    f = []
    print(np.shape(calculate_f(0, 12, W, wrapped_x)))
    f.append(calculate_f(0, 12, W, wrapped_x))
    f.append(calculate_f(24, 48, W, wrapped_x))
    print(np.shape(f))
    inv_cov_mat = get_covariance_matrix(f)
    attended_ear2 = np.array([attended_ear[0:12],attended_ear[24:48]])
    f_in_classes = group_by_class(f, attended_ear2)
    mean1 = calculate_mean(np.array(f_in_classes[0]))
    mean2 = calculate_mean(np.array(f_in_classes[1]))
    v_t, b = calculate_vt_b(inv_cov_mat, mean1, mean2)
    attended_ear2 = CSP.unwrap_cell_data(wrapped_attended_ear)[12:24]
    f = calculate_f(12, 24, W, wrapped_x)
    count = 0
    for i in range(12):
        D = calculate_D(v_t, b, f[i])
        if attended_ear2[i] != classify(D):
            count += 1
    print("Count:", count, (100 - (count * 100 / 12)), "% juist")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)

    # geval 3: verify 24-36
    f = [calculate_f(0, 24, W, wrapped_x), calculate_f(36, 48, W, wrapped_x)]
    f = np.array(f)
    inv_cov_mat = get_covariance_matrix(f)
    attended_ear2 = np.array([attended_ear[0:12], attended_ear[24:48]])
    f_in_classes = group_by_class(f, attended_ear2)
    mean1 = calculate_mean(np.array(f_in_classes[0]))
    mean2 = calculate_mean(np.array(f_in_classes[1]))
    v_t, b = calculate_vt_b(inv_cov_mat, mean1, mean2)
    attended_ear2 = CSP.unwrap_cell_data(wrapped_attended_ear)[24:36]
    f = calculate_f(24, 36, W, wrapped_x)
    count = 0
    for i in range(12):
        D = calculate_D(v_t, b, f[i])
        if attended_ear2[i] != classify(D):
            count += 1
    print("Count:", count, (100 - (count * 100 / 12)), "% juist")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)

    # geval 4: verificatie 36-48
    f = calculate_f(0, 36, W, wrapped_x)
    inv_cov_mat = get_covariance_matrix(f)
    f_in_classes = group_by_class(f, attended_ear[0:36])
    mean1 = calculate_mean(np.array(f_in_classes[0]))
    mean2 = calculate_mean(np.array(f_in_classes[1]))
    v_t, b = calculate_vt_b(inv_cov_mat, mean1, mean2)
    attended_ear2 = CSP.unwrap_cell_data(wrapped_attended_ear)[36:48]
    f = calculate_f(36, 48, W, wrapped_x)
    count = 0
    for i in range(12):
        D = calculate_D(v_t, b, f[i])
        if attended_ear2[i] != classify(D):
            count += 1
    print("Count:", count, (100 - (count * 100 / 12)), "% juist")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)