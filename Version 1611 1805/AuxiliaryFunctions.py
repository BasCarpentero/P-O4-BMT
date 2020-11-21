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
