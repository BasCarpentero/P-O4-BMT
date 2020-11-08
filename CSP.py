import numpy as np
from scipy.io import loadmat
import scipy.linalg as la


# A helper function to modify the given data, getting rid of Matlab's cell structures.
def unwrap_cell_data(cell_data):
    unwrapped_data = []
    for x in cell_data:
        unwrapped_data.append(x[0])
    return unwrapped_data


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


# A helper function to compute covariance matrices
def covariance_matrix(A):
    return np.dot(A, np.transpose(A)) / np.trace(np.dot(A, np.trace(A)))


# A helper function to compute composite covariance matrices given a list of classes.
def spatial_covariance_matrices(grouped_data):
    result = []
    for group in grouped_data.values():
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


if __name__ == "__main__":

    data = loadmat('dataSubject8.mat')
    wrapped_attended_ear = np.array(data.get('attendedEar'))
    attended_ear = unwrap_cell_data(wrapped_attended_ear)
    wrapped_EEG_data = np.array(data.get('eegTrials'))
    EEG_data = unwrap_cell_data(wrapped_EEG_data)

    grouped_data = group_by_class(EEG_data, attended_ear)
    class_covariances = spatial_covariance_matrices(grouped_data)
    test_W = CSP(class_covariances)
