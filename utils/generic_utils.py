import os
import sys

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

from utils import ucr_utils


class BaseClassicalModel(object):

    def __init__(self, name, **kwargs):
        """
        Base class of all Classical models to provide a unified interface
        for the training and evaluation engines.

        Args:
            name: Name of the classifier.
        """
        if name is None:
            name = self.__class__.__name__

        self.name = name

    def fit(self, X, y, training=True, **kwargs):
        """
        Unified method to train the classifer. Has access to mode parameter -
        `training` usually used by Neural Networks, to be used if required.

        Args:
            X: Training dataset.
            y: Training labels.
            training: Bool flag, whether training mode or
                evaluation mode. To be ignored by downstream
                subclasses (unless its a Neural Network under
                black-box treatment).
        """
        raise NotImplementedError()

    def predict(self, X, training=False, **kwargs):
        """
        Unified method to evaluate the classifier. Has access to mode parameter -
        `training` usually used by Neural Networks, to be used if required.

        Args:
            X: Test dataset.
            training: Bool flag, whether training mode or
                evaluation mode. To be ignored by downstream
                subclasses (unless its a Neural Network under
                black-box treatment).

        Returns:
            Predictions of the model.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """ Dispatch to the call(.) function """
        return self.call(*args, **kwargs)

    def call(self, x, training=False, **kwargs):
        """
        Perform evaluation when the classifier is called by the engine.
        Equivalent to `predict` with care taken to dispatch to numpy
        and reshape to 2d tensor.

        This is preferred when evaluation is required.

        Args:
            x: Dataset samples.
            training: Bool flag, whether training mode or
                evaluation mode. To be ignored by downstream
                subclasses (unless its a Neural Network under
                black-box treatment).

        Returns:
            Predictions of the model.
        """
        if hasattr(x, 'numpy'):  # is a tensor input
            x = x.numpy()  # make it a numpy ndarray

        if x.ndim > 2:  # reshape it to a 2d matrix for classical models
            x = x.reshape((x.shape[0], -1))

        return self.predict(x, training, **kwargs)

    def save(self, filepath):
        """ Save the class and its self contained dictionary of values """
        state = (self.__class__, self.__dict__)
        joblib.dump(state, filepath)

    @classmethod
    def restore(cls, filepath):
        """ Restore the class and its self contained dictionary of values """
        if os.path.exists(filepath):
            state = joblib.load(filepath)
            obj = cls.__new__(state[0])
            obj.__dict__.update(state[1])
            return obj
        else:
            raise FileNotFoundError("Model not found at %s" % filepath)


""" UTILITY FUNCTIONS """


# Obtained from keras.utils folder. Copied to remove unnecessary keras import.
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def checked_argmax(y, to_numpy=False):
    """
    Performs an argmax after checking if the input is either a tensor
    or a numpy matrix of rank 2 at least.

    Should be used in most cases for conformity throughout the
    codebase.

    Args:
        y: an numpy array or tensorflow tensor
        to_numpy: bool, flag to convert a tensor to a numpy array.

    Returns:
        an argmaxed array if possible, otherwise original array.
    """
    if hasattr(y, 'numpy'):
        if len(y.shape) > 1:
            y = tf.argmax(y, axis=-1)

        if to_numpy:
            return y.numpy()
        else:
            return y
    else:
        if y.ndim > 1:
            y = np.argmax(y, axis=-1)

        return y


def reranking(y, target, alpha):
    """
    Scales the activation of the target class, then normalizes to
    a probability distribution again.

    Args:
        y: The predicted label matrix of shape [N, C]
        target: integer id for selection of target class
        alpha: scaling factor for target class activations.
            Must be greater than 1.

    Returns:

    """
    # assert alpha > 1, "Alpha must be greater than 1."

    max_y = tf.reduce_max(y, axis=-1).numpy()

    if hasattr(y, 'numpy'):
        weighted_y = y.numpy()
    else:
        weighted_y = y

    weighted_y[:, target] = alpha * max_y

    weighted_y = tf.convert_to_tensor(weighted_y)

    result = weighted_y
    result = result / tf.reduce_sum(result, axis=-1, keepdims=True)  # normalize to probability distribution

    # print('** y normed ** ', result.shape, (np.mean(result.numpy()[:, target], )))

    return result


def rescaled_softmax(y, num_classes, tau=1.):
    """
    Scales the probability distribution of the input matrix by tau,
    prior to softmax being applied.

    Args:
        y: tensor / matrix of shape [N, C] or [N]
        num_classes: int, number of classes.
        tau: scaling temperature

    Returns:
        a scaled matrix of shape [N, C]
    """
    tau = float(tau)
    is_tensor = hasattr(y, 'numpy')

    if len(y.shape) > 1:
        # we are dealing with class probabilities of shape [N, C]
        y = tf.nn.softmax(y / tau, axis=-1)
    else:
        # we are dealing with class labels of shape [N], not class probabilities.
        # one hot encode the score and then scale.
        y = tf.one_hot(y, depth=num_classes)
        # y = tf.nn.softmax(y, axis=-1)

    if is_tensor:
        return y
    else:
        return y.numpy()


def target_accuracy(y_label, y_pred, target):
    """
    Computes the accuracy as well as num_adv of attack of the target class.

    Args:
        y_label: ground truth labels. Accepts one hot encodings or labels.
        y_pred: predicted labels. Accepts probabilities or labels.
        target: target class

    Returns:
        accuracy, target_rate
    """
    ground = checked_argmax(y_label, to_numpy=True)  # tf.argmax(y_label, axis=-1).numpy()
    predicted = checked_argmax(y_pred, to_numpy=True)  # tf.argmax(y_pred, axis=-1).numpy()
    accuracy = np.mean(np.equal(ground, predicted))

    non_target_idx = (ground != target)
    target_total = np.sum((predicted[non_target_idx] == target))
    target_rate = target_total / np.sum(non_target_idx)

    # Cases where non_target_idx is 0, so target_rate becomes nan
    if np.isnan(target_rate):
        target_rate = 1.  # 100% target num_adv for this batch

    return accuracy, target_rate


def enable_printing():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def disable_printing():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


""" DATASET UTILITIES """


def split_dataset(x_test, y_test, test_fraction=0.5):
    """
    Accepts an inputs dataset, and selects a portion of the test set to be
    the new train set for the adversarial model.

    Tries to extract such that number of samples in the new train set is
    same as the number of samples in the original train set.

    Uses class wise splitting to maintain counts from the test set.

    Args:
        X_test: numpy array
        y_test: numpy array

    Returns:
        (X_train, y_train), (X_test, y_test) with reduced number of samples
    """
    np.random.seed(0)
    num_classes = y_test.shape[-1]

    y_test = checked_argmax(y_test)

    test_labels, test_counts = np.unique(y_test, return_counts=True)
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    # Split the test set into adversarial train and adversarial test splits
    for label, max_cnt in zip(test_labels, test_counts):
        samples = x_test[y_test.flatten() == label]
        train_samples, test_samples = train_test_split(samples, test_size=test_fraction, random_state=0)

        train_cnt = len(train_samples)
        max_cnt = train_cnt

        X_train.append(train_samples[:max_cnt])
        Y_train.append([label] * max_cnt)

        X_test.append(test_samples)
        Y_test.append([label] * len(test_samples))

    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)

    from keras.utils import to_categorical
    y_train = to_categorical(np.concatenate(Y_train), num_classes)
    y_test = to_categorical(np.concatenate(Y_test), num_classes)

    print("\nSplitting test set into new train and test set !")
    print("X train = ", X_train.shape, "Y train : ", y_train.shape)
    print("X test = ", X_test.shape, "Y test : ", y_test.shape)

    return (X_train, y_train), (X_test, y_test)


def load_dataset(dataset_name='mnist'):
    """
    Resolves the provided dataset name into an image dataset provided
    by Keras dataset utils, or any ucr dataset (by id or name).

    Args:
        dataset_name: must be a string. Can be the dataset name
            for image datasets, and must be in the format
            'ucr/{id}' or 'ucr/{dataset_name}' for the ucr datasets.

    Returns:
        Image dataset : (X_train, y_train), (X_test, y_test)
        Time Series dataset : (X_train, y_train), (X_test, y_test), dictionary of dataset info
    """
    allowed_image_names = ['mnist', 'cifar10', 'cifar100', 'fmnist']

    if dataset_name in allowed_image_names:
        return load_image_dataset(dataset_name)

    ucr_split = dataset_name.split('/')
    if len(ucr_split) > 1 and ucr_split[0].lower() == 'ucr':
        # is a ucr dataset time series dataset
        id = -1

        try:
            id = int(ucr_split[-1])
        except ValueError:
            # assume it is a name of the time series dataset

            try:
                id = ucr_utils.DATASET_NAMES.index(ucr_split[-1].lower())
            except ValueError:
                print("Could not match %s to either id or name of dataset !" % (ucr_split[-1]))

        if id < 0:
            raise ValueError('Could not match %s to either id or name of dataset !' % (ucr_split[-1]))

        return load_ucr_dataset(id, normalize_timeseries=True)

    else:
        raise ValueError("Could not parse the provided dataset name : ", dataset_name)


def load_image_dataset(dataset_name='mnist'):
    """
    Loads the dataset by name.

    Args:
        dataset_name: string, either "mnist", "cifar10", "cifar100", "fmnist"

    Returns:
        (X_train, y_train), (X_test, y_test)
    """

    allowed_names = ['mnist', 'cifar10', 'cifar100', 'fmnist']

    if dataset_name not in allowed_names:
        raise ValueError("Dataset name provided is wrong. Must be one of ", allowed_names)

    #  print(directory)
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == 'fmnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    elif dataset_name == 'cifar100':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    else:
        raise ValueError('%s is not a valid dataset name. Available choices are : %s' % (
            dataset_name, str(allowed_names)
        ))

    if dataset_name in ['mnist', 'fmnist']:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.

    elif dataset_name in ['cifar10', 'cifar100']:
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.

    if dataset_name == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (X_test, y_test)


def load_ucr_dataset(index, normalize_timeseries=True, verbose=True):
    """
    Loads up a single dataset from the UCR Archive, normalizes it and
    provides information about it.

    Args:
        index: integer index of the dataset
        normalize_timeseries: Whether to normalize the time series per sample
        verbose: Whether to print log info about the dataset.

    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    assert index < len(ucr_utils.TRAIN_FILES)

    X_train, y_train, X_test, y_test, is_timeseries = ucr_utils.load_ucr_dataset_at(index,
                                                                                    normalize_timeseries,
                                                                                    verbose)

    num_classes = ucr_utils.NUM_CLASSES[index]

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    return (X_train, y_train), (X_test, y_test)


def prepare_dataset(X_train, y_train, X_test, y_test, batch_size, shuffle=True, device=None):
    """
    Constucts a train and test tf.Dataset for usage.

    Shuffles, repeats and batches the train set. Only batches the test set.
    Both datasets are pushed to the correct device for faster processing.

    Args:
        X_train: Train data
        y_train: Train label
        X_test: Test data
        y_test: Test label
        batch_size: batch size of the dataset
        shuffle: Whether to shuffle the train dataset or not
        device: string, 'cpu:0' or 'gpu:0' or some variant of that sort.

    Returns:
        two tf.Datasets, train and test datasets.
    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    if shuffle:
        train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(1000, seed=0))

    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.apply(tf.data.experimental.prefetch_to_device(device))

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.apply(tf.data.experimental.prefetch_to_device(device))

    return train_dataset, test_dataset


def plot_image_adversary(sequence, title, ax, remove_axisgrid=False,
                         xlabel=None, ylabel=None, legend=False,
                         imlabel=None, color=None, alpha=1.0):
    """
    Utility method for plotting a sequence.

    Args:
        sequence: A time series sequence.
        title: Title of the plot.
        ax: Axis of the subplot.
        remove_axisgrid: Whether to remove the axis grid.
        xlabel: Label of X axis.
        ylabel: Label of Y axis.
        legend: Whether to enable the legend.
        imlabel: Whether to label the sequence (for legend).
        color: Whetehr the sequence should be of certain color.
        alpha: Alpha value of the sequence.

    Returns:
        Note, this method does not automatically call plt.show().

        This is to allow multiple subplots to borrow the same call
        without immediate plotting.

        Therefore, do not forget to call `plt.show()` at the end.
    """
    if remove_axisgrid:
        ax.axis('off')

    if sequence.ndim > 3:
        raise ValueError("Data provided cannot be more than rank 3 tensor.")
    else:
        ax.plot(sequence.flatten(), color=color, alpha=alpha, label=imlabel)

        if title is not None:
            ax.set_title(str(title), fontsize=20)

    if xlabel is not None:
        ax.get_xaxis().set_visible(True)
        ax.set_xlabel(str(xlabel))

    if ylabel is not None:
        ax.get_yaxis().set_visible(True)
        ax.set_ylabel(str(ylabel))

    if legend:
        ax.legend(loc='upper right')
