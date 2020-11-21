"""
This script contains the code for computing a common spatial pattern filter.
"""

import numpy as np
import scipy.linalg as la

import AuxiliaryFunctions


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
    return {"class_one": class_one, "class_two": class_two}


# A helper function to compute composite covariance matrices given a list of classes.
def spatial_covariance_matrices(grouped_data):
    result = []
    for group in grouped_data.values():
        # Initialize an empty matrix.
        spatial_covariance = AuxiliaryFunctions.covariance_matrix(group[0]) * 0
        count = 0
        for sample in group:
            count += 1
            spatial_covariance += AuxiliaryFunctions.covariance_matrix(sample)
        result.append(spatial_covariance / count)
    return result


# Computes a spatial filter
def CSP(class_covariances):
    # Solve the generalized eigenvalue problem resulting in eigenvalues and corresponding eigenvectors and
    # sort them in descending order.
    eigenvalues, eigenvectors = la.eigh(class_covariances[0], class_covariances[1])
    ascending_eigenvalues = np.argsort(eigenvalues)
    descending_eigenvalues = ascending_eigenvalues[::-1]
    eigenvalues = eigenvalues[descending_eigenvalues]
    eigenvectors = eigenvectors[:, descending_eigenvalues]
    return eigenvectors
