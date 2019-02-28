import numpy as np
import pandas as pd
import os


DATASET_NAMES = None
TRAIN_FILES = None
TEST_FILES = None
NUM_CLASSES = None
NUM_TIMESTEPS = None


def _populate_information():
    """
    Populates the global information about the datasets into lists
    so that they can be utilized by lower stages.
    """
    utils_path = '../utils'

    if not os.path.exists(utils_path):
        utils_path = utils_path[1:]

    df = pd.read_csv(utils_path + '/UCRDataSummary.csv', header=0, encoding='latin-1')
    names = df.Name.values

    global DATASET_NAMES

    if DATASET_NAMES is None:
        DATASET_NAMES = [name.lower() for name in names]

    global TRAIN_FILES, TEST_FILES

    # already loaded check
    if TRAIN_FILES is not None and TEST_FILES is not None:
        return

    if TRAIN_FILES is None:
        TRAIN_FILES = []

    if TEST_FILES is None:
        TEST_FILES = []

    basepath = '../data'

    if not os.path.exists(basepath):
        basepath = basepath[1:]

    for name in names:
        if os.path.exists(basepath + "/%s_TRAIN" % (name)):
            TRAIN_FILES.append(basepath + "/%s_TRAIN" % (name))
        else:
            print("Dataset not found ! ", name)

        if os.path.exists(basepath + "/%s_TEST" % (name)):
            TEST_FILES.append(basepath + "/%s_TEST" % (name))
        else:
            print("Dataset not found ! ", name)

    classes = df.Class.values
    lengths = df.Length.values

    global NUM_CLASSES, NUM_TIMESTEPS

    if NUM_CLASSES is not None and NUM_TIMESTEPS is not None:
        return

    if NUM_CLASSES is None:
        NUM_CLASSES = classes

    if NUM_TIMESTEPS is None:
        NUM_TIMESTEPS = lengths


def load_ucr_dataset_at(index, normalize_timeseries=False, verbose=True):
    """
    Loads a Univaraite UCR Dataset indexed by UCRDataSummary.csv.

    Args:
        index: Integer index.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
        verbose: Whether to describe the dataset being loaded.

    Returns:
        A tuple of shape (X_train, y_train, X_test, y_test, is_timeseries).
        For legacy reasons, is_timeseries is always True.
    """
    global TRAIN_FILES, TEST_FILES

    assert index < len(TRAIN_FILES), "Index invalid. Could not load dataset at %d" % index
    if verbose:
        print("Loading train / test dataset : ", TRAIN_FILES[index], TEST_FILES[index])

    if os.path.exists(TRAIN_FILES[index]):
        df = pd.read_csv(TRAIN_FILES[index], header=None, encoding='latin-1')

    elif os.path.exists(TRAIN_FILES[index][1:]):
        df = pd.read_csv(TRAIN_FILES[index][1:], header=None, encoding='latin-1')

    else:
        raise FileNotFoundError('File %s not found!' % (TRAIN_FILES[index]))

    is_timeseries = True  # assume all input data is univariate time series

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    if not is_timeseries:
        data_idx = df.columns[1:]
        min_val = min(df.loc[:, data_idx].min())
        if min_val == 0:
            df.loc[:, data_idx] += 1

    # fill all missing columns with 0
    df.fillna(0, inplace=True)

    # cast all data into integer (int32)
    if not is_timeseries:
        df[df.columns] = df[df.columns].astype(np.int32)

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    y_train = df[[0]].values
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_train = df.values

    if is_timeseries:
        X_train = X_train[:, :, np.newaxis]
        # scale the values
        if normalize_timeseries:
            normalize_timeseries = int(normalize_timeseries)

            if normalize_timeseries == 2:
                X_train_mean = X_train.mean()
                X_train_std = X_train.std()
                X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

            else:
                X_train_mean = X_train.mean(axis=1, keepdims=True)
                X_train_std = X_train.std(axis=1, keepdims=True)
                X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if verbose: print("Finished loading train dataset..")

    if os.path.exists(TEST_FILES[index]):
        df = pd.read_csv(TEST_FILES[index], header=None, encoding='latin-1')

    elif os.path.exists(TEST_FILES[index][1:]):
        df = pd.read_csv(TEST_FILES[index][1:], header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (TEST_FILES[index]))

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    if not is_timeseries:
        data_idx = df.columns[1:]
        min_val = min(df.loc[:, data_idx].min())
        if min_val == 0:
            df.loc[:, data_idx] += 1

    # fill all missing columns with 0
    df.fillna(0, inplace=True)

    # cast all data into integer (int32)
    if not is_timeseries:
        df[df.columns] = df[df.columns].astype(np.int32)

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    y_test = df[[0]].values
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_test = df.values

    if is_timeseries:
        X_test = X_test[:, :, np.newaxis]
        # scale the values
        if normalize_timeseries:
            normalize_timeseries = int(normalize_timeseries)

            if normalize_timeseries == 2:
                X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)
            else:
                X_test_mean = X_test.mean(axis=1, keepdims=True)
                X_test_std = X_test.std(axis=1, keepdims=True)
                X_test = (X_test - X_test_mean) / (X_test_std + 1e-8)

    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[1])
        print("X Train mean/std : ", X_train.mean(), X_train.std())
        print("X Test mean/std : ", X_test.mean(), X_test.std())

    return X_train, y_train, X_test, y_test, is_timeseries


def calculate_dataset_metrics(X_train):
    """
    Calculates the dataset metrics used for model building and evaluation.

    Args:
        X_train: The training dataset.

    Returns:
        A tuple of (None, sequence_length). None is for legacy
        purposes.
    """
    is_timeseries = len(X_train.shape) == 3
    if is_timeseries:
        # timeseries dataset
        max_sequence_length = X_train.shape[-1]
        max_nb_words = None
    else:
        # transformed dataset
        max_sequence_length = X_train.shape[-1]
        max_nb_words = np.amax(X_train) + 1

    return max_nb_words, max_sequence_length


# load the global tables
_populate_information()


if __name__ == '__main__':
    _populate_information()

    ids = [86, 87, 88, 104, 105, 115, 116, 125]

    for id in ids:
        X_train, y_train, X_test, y_test, _ = load_ucr_dataset_at(id, verbose=False)

        print("Dataset ID : ", id, "Name : ", DATASET_NAMES[id], "Num timesteps : ", X_train.shape[1])
