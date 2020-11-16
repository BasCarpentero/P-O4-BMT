import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import CSP
from scipy.io import loadmat
import matplotlib.pyplot as plt



# Returns the log-energy vector computed with T samples given the CSP-filtered data y.
def energy(y, decisionWindow):
    outputEnergyVector = np.zeros(len(y))
    for i in range(decisionWindow):
        for j in range(0, len(y)):
            outputEnergyVector[j] += (y[j][i])**2
    return np.log(outputEnergyVector)


def calculate_f(testMinutes, W, x):
    f = []
    for i in testMinutes:
        # x is a C x A matrix with C = #channels and A = #time instances
        # W is a C x K matrix with C = #channels and K = # spatial filters
        xi = x[i][0].T
        y = np.dot(W.T, xi)
        T = 7190
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


def classify(v_t, b, f):
    if np.dot(v_t, f) + b > 0:
        return 1
    else:
        return 2


if __name__ == "__main__":
    # data = loadmat('/Users/ogppr/Documents/dataSubject8.mat')
    data = loadmat('dataSubject8.mat')

    # Select the corresponding attended ear results for each test case.
    wrapped_attended_ear = np.array(data.get('attendedEar'))
    attended_ear = CSP.unwrap_cell_data(wrapped_attended_ear)
    # attended_ear_2 = np.delete(attended_ear, np.s_[24-36], axis=0)
    # attended_ear_3 = np.delete(attended_ear, np.s_[12-24], axis=0)
    # attended_ear_4 = np.delete(attended_ear, np.s_[0-12], axis=0)

    # Select the test data columns for each case.
    wrapped_EEG_data = np.array(data.get('eegTrials'))
    wrapped_x = np.array(data.get('eegTrials'))
    EEG_data = CSP.unwrap_cell_data(wrapped_EEG_data)
    # EEG_data_2 = np.delete(EEG_data, np.s_[24-36], axis=0)
    # EEG_data_3 = np.delete(EEG_data, np.s_[12-24], axis=0)
    # EEG_data_4 = np.delete(EEG_data, np.s_[0-12], axis=0)

    # grouped_data = CSP.group_by_class(EEG_data, attended_ear)
    # class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    # W = CSP.CSP(class_covariances)

    # case 1: verification 36-48
    #training
    EEG_data_1 = EEG_data[0:36]
    attended_ear_1 = attended_ear[:36]
    grouped_data = CSP.group_by_class(EEG_data_1, attended_ear_1)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    W = CSP.CSP(class_covariances)

    trainMinutes = []
    for i in range(0, 36):
        trainMinutes.append(i)
    f = calculate_f(trainMinutes, W, wrapped_x) #dimensies: 36x24

    lda = LDA()
    lda.fit(f,attended_ear_1)

    # inv_cov_mat = calculate_covariance_matrix(f)
    # f_in_classes = group_by_class(f, attended_ear_1)
    # mean1 = calculate_mean(np.array(f_in_classes[0]))
    # mean2 = calculate_mean(np.array(f_in_classes[1]))
    # v_t, b = calculate_vt_b(inv_cov_mat, mean1, mean2)


    testMinutes = []
    for i in range(0, 12):
        testMinutes.append(i)

    #verification
    f = calculate_f(testMinutes, W, wrapped_x)
    resultaat = lda.predict(f)
    count = 0
    for i in range(12):
        print(attended_ear[36+i], resultaat[i])
        if attended_ear[36+i] != resultaat[i]:
            count += 1
    print("Count:", count, (100 - (count * 100 / 12)), "% juist")  # Aantal verkeerd voorspelde minuten (veel te hoog!!)


    # case 2: verification 0-12
    EEG_data_2 = EEG_data[12:]
    attended_ear_2 = attended_ear[12:]
    grouped_data = CSP.group_by_class(EEG_data_2, attended_ear_2)
    class_covariances = CSP.spatial_covariance_matrices(grouped_data)
    W = CSP.CSP(class_covariances)
    trainMinutes = []
    for i in range(12, 48):
        trainMinutes.append(i)
    f = calculate_f(trainMinutes, W, wrapped_x)
    lda = LDA()
    lda.fit(f,attended_ear_2)

    # plt.figure()
    # plt.scatter(f[0],f[1],color='red')
    # plt.scatter(f[2], f[3], color='blue')
    # plt.scatter(f[4],f[5],color='green')
    # plt.show()
    # plt.close()
    # inv_cov_mat = calculate_covariance_matrix(f)
    # f_in_classes = group_by_class(f, attended_ear[12:48])
    # mean1 = calculate_mean(np.array(f_in_classes[0]))
    # mean2 = calculate_mean(np.array(f_in_classes[1]))
    # v_t, b = calculate_vt_b(inv_cov_mat, mean1, mean2)

    testMinutes = []
    for i in range(0, 12):
        testMinutes.append(i)
    f = calculate_f(testMinutes, W, wrapped_x)
    resultaat = lda.predict(f)
    count = 0
    for i in range(12):
        print(attended_ear[i],resultaat[i])
        if attended_ear[i] != resultaat[i]:
            count += 1
    print("Count:", count, (100 - (count*100/12)), "% juist")