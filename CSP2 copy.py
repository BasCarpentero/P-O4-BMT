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
            class_one.append(samples[x])
        else:
            class_two.append(samples[x])
    if len(class_one) > 0:
        if len(class_two) > 0:
            return {"class_one": class_one, "class_two": class_two}
        else:
            return {"class_one": class_one}
    else:
        return {"class_two": class_two}


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
def CSP(the_class_covariances):
    composite_covariance = the_class_covariances[0] * 0
    for x in the_class_covariances:
        composite_covariance += x

    # Compute the composite covariance's eigenvalues E and corresponding eigenvectors V and
    # sort them in descending order.
    E, V = la.eig(composite_covariance)
    ascending_eigenvalues = np.argsort(E)
    descending_eigenvalues = ascending_eigenvalues[::-1]
    E = E[descending_eigenvalues]
    V = V[:, descending_eigenvalues]

    # Calculate the whitening transformation.
    P = np.dot(np.sqrt(la.inv(np.diag(E))), np.transpose(V))

    # Transform the mean class variances
    for i in range(len(the_class_covariances)):
        the_class_covariances[i] = np.dot(P, np.dot(the_class_covariances[i], np.transpose(P)))
    the_class_covariances = the_class_covariances[0]

    # Find and sort the generalized eigenvalues and eigenvector
    E_gen, V_gen = la.eig(the_class_covariances)
    ascending_eigenvalues_gen = np.argsort(E_gen)
    descending_eigenvalues_gen = ascending_eigenvalues_gen[::-1]
    E_gen = E_gen[descending_eigenvalues_gen]
    V_gen = V_gen[:, descending_eigenvalues_gen]

    # The projection matrix is given as:
    W = np.dot(np.transpose(V_gen), P)
    return W


if __name__ == "__main__":

    data = loadmat('dataSubject8.mat')
    wrapped_attended_ear = np.array(data.get('attendedEar'))[:2]
    # attended_ear = wrapped_attended_ear[:2]
    attended_ear = unwrap_cell_data(wrapped_attended_ear)
    wrapped_EEG_data = np.array(data.get('eegTrials'))[:2]
    # EEG_data = wrapped_EEG_data[:2]
    EEG_data = unwrap_cell_data(wrapped_EEG_data)

    grouped_data = group_by_class(EEG_data, attended_ear)
    class_covariances = spatial_covariance_matrices(grouped_data)
    print(CSP(class_covariances))
