import numpy as np
import scipy.linalg as la


# A helper function to compute covariance matrices
def covariance_matrix(A):
    return np.dot(A, np.transpose(A)) / np.trace(np.dot(A, np.trace(A)))


# A helper function to compute composite covariance matrices given a list of classes.
def spatial_covariance_matrices(grouped_data):
    result = []
    for group in grouped_data:
        # Initialize an empty matrix.
        spatial_covariance = covariance_matrix(group[0]) * 0
        count = 0
        for sample in group:
            count += 1
            spatial_covariance += covariance_matrix(sample)
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
