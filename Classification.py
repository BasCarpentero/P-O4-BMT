import numpy as np
import pandas as pd
import random
import sys, itertools
import matplotlib.pyplot as plt
import scipy.stats
from scipy.io import loadmat
import CSP
import FeatureExtraction


def get_data():
    # import feature vectors and attended speakers
    # features will be formatted as column vectors with n rows for n dimensions
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

    t = 7170
    return FeatureExtraction.createFeature(y, t)  # f


# This class transform a given ... of feature vectors to one single dimension, minimizing the class variances and
# maximizing the distance between classes through their means.
class Fischer_linear_discriminant:
    def __init__(self, data, dimension):
        self.data = data # Data is given as a list of feature vectors
        self.dimension = dimension # The dimension of the feature vector
        self.dimension = len(self.data[0]) - 1
        self.set_group()
        self.calculate_means()
        self.calculate_covariances()
        self.calculate_eigenvalues()
        self.transform()
        self.get_classes()

    # Step 1: Group the features vectors into their classes (speaker 1 or speaker 2).
    def set_group(self):
        class_one = []
        class_two = []
        classes = Fischer_linear_discriminant.get_classes()
        for x in classes:
            if x == 1:
                class_one.append(np.transpose(self.data[x]))
            else:
                class_two.append(np.transpose(self.data[x]))
        self.grouped_features = np.array([class_one, class_two])

    def get_classes(self):
        data = loadmat('dataSubject8.mat')
        wrapped_attended_ear = np.array(data.get('attendedEar'))[:36]
        attended_ear = CSP.unwrap_cell_data(wrapped_attended_ear)
        return attended_ear

    # Step 2: Calculate the overall mean and per class means.
    def calculate_means(self):
        self.class_mean = {}
        self.overall_mean = np.array([0. for x in range(self.dimension)])
        for i in self.grouped_features:
            self.class_mean[i] = np.array([0. for x in range(self.dimension)])
            for j in self.grouped_features[i]:
                for k in range(len(j)):
                    self.class_mean[i][k] += j[k]
                    self.overall_mean[k] += j[k]

        for i in self.class_mean:
            for j in range(len(self.class_mean[i])):
                self.class_mean[i][j] /= len(self.grouped_features[i])

        for i in range(len(self.overall_mean)):
            self.overall_mean[i] /= len(self.training_data)

    # Step 3: Calculate between-class and within-class covariance matrices.
    def calculate_covariances(self):
        self.between_covariance = np.zeros((self.dimension, self.dimension))
        for i in self.class_mean:
            group_size = len(self.grouped_features[i])
            mean_difference = np.array([self.class_mean[i] - self.overall_mean])
            mean_difference_transpose = mean_difference.transpose()
            group_term = mean_difference_transpose.dot(mean_difference)
            self.between_covariance += group_term

        self.within_covariance = np.zeros((self.dimension, self.dimension))
        for i in self.class_mean:
            class_mean = np.array([self.class_mean])
            for j in self.grouped_features[i]:
                feature = np.array(j)
                feature_minus_mean = np.array([feature - class_mean])
                feature_minus_mean_transpose = feature_minus_mean.transpose()
                self.within_covariance += feature_minus_mean_transpose.dot(feature_minus_mean)

    # Step 4: Calculate the eigenvalues of X and get n eigenvectors for n desired dimensions
    def calculate_eigenvalues(self):
        covariance_product = np.dot(np.linalg.pinv(self.within_covariance), self.between_covariance) # Solve the eigenvalue problem inverse(within_covariance) * between_covariance * w = Lagrange_multiplier * w
        eigenvalues, eigenvectors = np.linalg.eig(covariance_product)
        eigenvalue_list = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eigenvalue_list = sorted(eigenvalue_list, key=lambda x: x[0], reverse=True) # A sorted list form highest to lowest eigenvalue with corresponding eigenvectors
        self.W = np.array([eigenvalue_list[i][1]] for i in range(self.dimension))

    # Step 5: Use the eigenvectors of 4 to construct Y to transform the training data
    def construct_y(self):
        self.Y = np.dot(self.data, self.W)
    # Step 6: Apply the transformation
    # Step 7: Classify by the calculated threshold
