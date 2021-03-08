import numpy as np


# Modifies the given data, getting rid of Matlab's cell structures.
def unwrap_cell_data(cell_data):
    unwrapped_data = []
    for x in cell_data:
        unwrapped_data.append(x[0])
    return unwrapped_data


# A helper function to compute covariance matrices
def covariance_matrix(A):
    return np.cov(A)


# A helper function to classify given samples into their classes, returns a dictionary with keys 'class_x' and a list
# of corresponding samples as values.
def group_by_class(samples, sample_classes):
    class_one = []
    class_two = []
    for i in range(len(samples)):
        if sample_classes[i] == 1:
            class_one.append(np.transpose(samples[i]))
        else:
            class_two.append(np.transpose(samples[i]))
    return np.array([class_one, class_two])

def lwcov(X):
    nobs,nvar = X.shape[0],X.shape[1]
    X = X-np.average(X,axis=0)
    S = (1/(nobs-1))*np.matmul(np.transpose(X),X)
    m = np.trace(S)/nvar
    d2 = np.square(np.linalg.norm(S-m*np.eye(nvar)))/nvar
    rownorms = np.sum(np.square(X),axis=1)
    term11= np.matmul(np.transpose(X),np.multiply(np.transpose(X),rownorms))
    term12 = -2*np.matmul(S,(np.matmul(np.transpose(X),X)))
    term22 = nobs*np.matmul(S,np.transpose(S))
    b2 = np.trace(term11+term12+term22)/(nvar*nobs^2)
    a2 = d2 - b2
    Sr = b2/d2 *m *np.eye(nvar)+a2/d2*S

    return Sr