import numpy as np


# Modifies the given data, getting rid of Matlab's cell structures.
def unwrap_cell_data(cell_data):
    unwrapped_data = []
    for x in cell_data:
        unwrapped_data.append(x[0])
    return unwrapped_data


# A helper function to compute covariance matrices
def covariance_matrix(A):
    return np.dot(A, np.transpose(A)) / np.trace(np.dot(A, np.trace(A)))


# A helper function to classify given samples into their classes, returns a dictionary with keys 'class_x' and a list
# of corresponding samples as values.
def group_by_class(samples, sample_classes):
    class_one = []
    class_two = []
    for x in sample_classes:
        if x == 1:
            class_one.append(np.transpose(samples[x]))
        else:
            class_two.append(np.transpose(samples[x]))
    return np.array([class_one, class_two])#, dtype=object)