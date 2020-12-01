"""
This script contains the code for computing a common spatial pattern filter.
"""

import numpy as np
import scipy.linalg as la

import AuxiliaryFunctions

# A helper function to compute composite covariance matrices given a list of classes.
def spatial_covariance_matrices(grouped_data):
    result = []
    for group in grouped_data:
        # Initialize an empty matrix.
        spatial_covariance = AuxiliaryFunctions.covariance_matrix(group[0]) * 0
        count = 0
        for sample in group:
            count += 1
            spatial_covariance += AuxiliaryFunctions.covariance_matrix(sample)
        result.append(spatial_covariance / count)
    return result


# Computes a spatial filter
def CSP(class_covariances, size):
    # Solve the generalized eigenvalue problem resulting in eigenvalues and corresponding eigenvectors and
    # sort them in descending order.
    eigenvalues, eigenvectors = la.eigh(class_covariances[0], class_covariances[1])
    id_descending = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, id_descending]
    eigenvectors_begin = np.array(eigenvectors[:, :int(size/2)])
    eigenvectors_end = np.array(eigenvectors[:, int(np.shape(eigenvectors)[1]-size/2):])
    eigenvectors = np.concatenate((eigenvectors_begin, eigenvectors_end), axis=1)
    return eigenvectors
