import numpy as np
from numba import jit, prange

from scipy.stats import mode
from sklearn.metrics import accuracy_score

try:
    from IPython.core.display import HTML, Javascript, display
except ImportError:
    pass


__all__ = ['dtw_distance', 'KnnDTW']


@jit(nopython=True, parallel=True, nogil=True)
def dtw_distance(dataset1, dataset2):
    n1 = dataset1.shape[0]
    n2 = dataset2.shape[0]
    dist = np.empty((n1, n2), dtype=np.float64)

    for i in prange(n1):
        for j in prange(n2):
            dist[i][j] = _dtw_distance(dataset1[i], dataset2[j])

    return dist


@jit(nopython=True, cache=True)
def _dtw_distance(series1, series2):
    """
    Returns the DTW similarity distance between two 2-D
    timeseries numpy arrays.

    Arguments
    ---------
    series1, series2 : array of shape [n_samples, n_timepoints]
        Two arrays containing n_samples of timeseries data
        whose DTW distance between each sample of A and B
        will be compared

    Returns
    -------
    DTW distance between A and B
    """
    l1 = series1.shape[0]
    l2 = series2.shape[0]
    E = np.empty((l1, l2))

    # Fill First Cell
    v = series1[0] - series2[0]
    E[0][0] = v * v

    # Fill First Column
    for i in range(1, l1):
        v = series1[i] - series2[0]
        E[i][0] = E[i - 1][0] + v * v

    # Fill First Row
    for i in range(1, l2):
        v = series1[0] - series2[i]
        E[0][i] = E[0][i - 1] + v * v

    for i in range(1, l1):
        for j in range(1, l2):
            v = series1[i] - series2[j]
            v = v * v

            v1 = E[i - 1][j]
            v2 = E[i - 1][j - 1]
            v3 = E[i][j - 1]

            if v1 <= v2 and v1 <= v3:
                E[i][j] = v1 + v
            elif v2 <= v1 and v2 <= v3:
                E[i][j] = v2 + v
            else:
                E[i][j] = v3 + v

    return np.sqrt(E[-1][-1])


# Modified from https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping
class KnnDTW(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 1)
        Number of neighbors to use by default for KNN

    window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function

    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        """Fit the model using x as training data and y as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        y : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

        self.x = np.copy(x)
        self.y = np.copy(y)

    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """
        dm = dtw_distance(x, y)
        return dm

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """
        np.random.seed(0)
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.y[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()

    def evaluate(self, x, y):
        """
        Predict the class labels or probability estimates for
        the provided data and then evaluates the accuracy score.

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

          y : array of shape [n_samples]
              Array containing the labels of the testing dataset to be classified

        Returns
        -------
          1 floating point value representing the accuracy of the classifier
        """
        # Predict the labels and the probabilities
        pred_probas, pred_labels = self.predict(x)

        # Ensure labels are integers
        y = y.astype('int32')
        pred_labels = pred_labels.astype('int32')

        # Compute accuracy measure
        accuracy = accuracy_score(y, pred_labels)
        return accuracy

    def predict_proba(self, x):
        """Predict the class label probability estimates for
        the provided data

        > Implementation of Soft 1-NN DTW representation (Algorithm 1),
        from the paper "Adversarial Attacks on Time Series" [1]

        Arguments
        ---------
            x : array of shape [n_samples, n_timepoints]
                Array containing the testing data set to be classified

        Returns
        -------
            2 arrays representing:
                (1) the predicted class probabilities
                (2) the knn labels

        References
        -------
        [1] [Adversarial Attacks on Time Series]()
        """
        if self.n_neighbors != 1:
            raise RuntimeError("Soft-1NN transformation can only be applied to the K=1 case !")

        np.random.seed(0)
        dm = self._dist_matrix(x, self.x)

        dm = -dm

        classes = np.unique(self.y)
        class_dm = []

        # Partition distance matrix by class
        for i, cls in enumerate(classes):
            idx = np.argwhere(self.y == cls)[:, 0]
            cls_dm = dm[:, idx]  # [N_test, N_train_c]

            # Take minimum distance vector
            cls_dm = np.max(cls_dm, axis=-1)  # [N_test,]

            class_dm.append([cls_dm])

        class_dm = np.concatenate(class_dm, axis=0)  # [C, N_test]
        class_dm = class_dm.transpose()  # [N_test, C]

        class_dm_exp = np.exp(class_dm - class_dm.max())
        class_dm = class_dm_exp / np.sum(class_dm_exp, axis=-1, keepdims=True)

        knn_labels = np.argmax(class_dm, axis=-1)

        return class_dm, knn_labels
