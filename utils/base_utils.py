import os
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

import utils.generic_utils as generic_utils


def _compute_preds_loss_grad(model, x, y):
    """
    Computes the loss and gradients of a trainable model.

    Args:
        model: A tf.keras.Model
        x: training data
        y: ground truth labels

    Returns:
        a tuple (preds, loss, gradients) of lists.
    """
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y, y_pred)

    gradients = tape.gradient(loss, model.variables)
    return y_pred, loss, gradients


def train_base(model_fn, dataset_name, epochs=300, batchsize=128, lr=1e-3, model_name=None, device=None):
    """
    Trains the base classifier (aka the attacked classifier).

    Trains only Neural Networks.

    Args:
        model_fn: A callable function that returns a tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        epochs: Number of epochs to train the model.
        batchsize: Size of each batch.
        lr: Initial learning rate.
        model_name: Name of the model being built.
        device: Device to place the model on.

    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    # Load the dataset
    (X_train, y_train), (X_test, y_test) = generic_utils.load_dataset(dataset_name)

    num_classes = y_train.shape[-1]
    num_train_batches = X_train.shape[0] // batchsize + int(X_train.shape[0] % batchsize != 0)
    num_test_batches = X_test.shape[0] // batchsize + int(X_test.shape[0] % batchsize != 0)

    # build the datasets
    train_dataset, test_dataset = generic_utils.prepare_dataset(X_train, y_train,
                                                                X_test, y_test,
                                                                batch_size=batchsize,
                                                                device=device)

    # construct the model on the correct device
    with tf.device(device):
        if model_name is not None:
            model = model_fn(num_classes, name=model_name)  # type: tf.keras.Model
        else:
            model = model_fn(num_classes)  # type: tf.keras.Model

    lr_schedule = tf.train.exponential_decay(lr, tf.train.get_or_create_global_step(),
                                             decay_steps=num_train_batches, decay_rate=0.99,
                                             staircase=True)

    optimizer = tf.train.AdamOptimizer(lr_schedule)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer,
                                     global_step=tf.train.get_or_create_global_step())

    model_name = model.name if model_name is None else model_name
    basepath = 'weights/%s/%s/' % (dataset_name, model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + model_name

    best_loss = np.inf

    print()

    # train loop
    for epoch_id in range(epochs):
        train_loss = tfe.metrics.Mean()
        test_loss = tfe.metrics.Mean()

        train_acc = tfe.metrics.Mean()
        test_acc = tfe.metrics.Mean()

        with tqdm(train_dataset,
                  desc="Epoch %d / %d: " % (epoch_id + 1, epochs),
                  total=num_train_batches, unit=' samples') as iterator:

            for train_iter, (x, y) in enumerate(iterator):
                y_preds, loss_vals, grads = _compute_preds_loss_grad(model, x, y)
                loss_val = tf.reduce_mean(loss_vals)

                # update model weights
                grad_vars = zip(grads, model.variables)
                optimizer.apply_gradients(grad_vars, tf.train.get_or_create_global_step())

                # compute and update training target_accuracy
                acc_val = tf.keras.metrics.categorical_accuracy(y, y_preds)

                train_loss(loss_val)
                train_acc(acc_val)

                if train_iter >= num_train_batches:
                    break

        with tqdm(test_dataset, desc='Evaluating',
                  total=num_test_batches, unit=' samples') as iterator:
            for x, y in iterator:
                y_preds = model(x, training=False)
                loss_val = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_preds))

                # compute and update the test target_accuracy
                acc_val = tf.keras.metrics.categorical_accuracy(y, y_preds)

                test_loss(loss_val)
                test_acc(acc_val)

        print("\nEpoch %d: Train Loss = %0.5f | Train Acc = %0.6f | Test Loss = %0.5f | Test Acc = %0.6f" % (
            epoch_id + 1, train_loss.result(), train_acc.result(), test_loss.result(), test_acc.result()
        ))

        train_loss_val = train_loss.result()
        if best_loss > train_loss_val:
            print("Saving weights as training loss improved from %0.5f to %0.5f!" % (best_loss, train_loss_val))
            print()

            best_loss = train_loss_val

            checkpoint.write(checkpoint_path)

    print("\n\n")
    print("Finished training !")


def evaluate_model(model_fn, dataset_name, batchsize=128, model_name=None, device=None):
    """
    Evaluates the base classifier (aka the attacked classifier).

    Evaluates only Neural Networks.

    Args:
        model_fn: A callable function that returns a tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        batchsize: Size of each batch.
        model_name: Name of the model being built.
        device: Device to place the model on.

    Returns:
        (test_loss, test_acc)
    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    # Load the dataset
    (X_train, y_train), (X_test, y_test) = generic_utils.load_dataset(dataset_name)
    num_classes = y_train.shape[-1]

    num_test_batches = X_test.shape[0] // batchsize + int(X_test.shape[0] % batchsize != 0)

    # build the datasets
    train_dataset, test_dataset = generic_utils.prepare_dataset(X_train, y_train,
                                                                X_test, y_test,
                                                                batch_size=batchsize,
                                                                device=device)

    # construct the model on the correct device
    with tf.device(device):
        if model_name is not None:
            model = model_fn(num_classes, name=model_name)  # type: tf.keras.Model
        else:
            model = model_fn(num_classes)  # type: tf.keras.Model

    optimizer = tf.train.AdamOptimizer()

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer,
                                     global_step=tf.train.get_or_create_global_step())

    model_name = model.name
    basepath = 'weights/%s/%s/' % (dataset_name, model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + model_name

    # restore the parameters that were saved
    checkpoint.restore(checkpoint_path)

    # train loop
    test_loss = tfe.metrics.Mean()
    test_acc = tfe.metrics.Mean()

    with tqdm(test_dataset, desc='Evaluating',
              total=num_test_batches, unit=' samples') as iterator:
        for x, y in iterator:
            y_preds = model(x, training=False)
            loss_val = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_preds))

            # compute and update the test target_accuracy
            acc_val = tf.keras.metrics.categorical_accuracy(y, y_preds)

            test_loss(loss_val)
            test_acc(acc_val)
    print("\nTest Loss = %0.5f | Test Acc = %0.6f" % (test_loss.result(), test_acc.result()))

    return test_loss.result(), test_acc.result()


def train_classical_model(model_fn, dataset_name, model_name=None, evaluate=True):
    """
    Trains the base classifier (aka the attacked classifier).

    Trains only subclasses of BaseClassicalModel.

    Args:
        model_fn: A callable function that returns a subclassed BaseClassicalModel.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        model_name: Name of the model being built.
        evaluate: Whether to evaluate on the test set after training.
            This is only for observation, and takes significant time
            so it should be avoided unless absolutely necessary.
    """
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = generic_utils.load_dataset(dataset_name)
    num_classes = y_train.shape[-1]

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    # construct the model on the correct device
    if model_name is not None:
        model = model_fn(num_classes, name=model_name)  # type: generic_utils.BaseClassicalModel
    else:
        model = model_fn(num_classes)  # type: generic_utils.BaseClassicalModel

    model_name = model.name if model_name is None else model_name
    basepath = 'weights/%s/%s/' % (dataset_name, model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath)

    checkpoint_path = basepath + model_name + '.pkl'

    print()

    # train loop
    model.fit(X_train, y_train)

    # Save the model
    model.save(checkpoint_path)

    if evaluate:
        # Evaluate on train set once
        y_pred = model(X_train)

        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)

        train_accuracy = accuracy_score(y_train, y_pred)

        # Evaluate on test set once
        y_pred = model(X_test)

        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)

        test_accuracy = accuracy_score(y_test, y_pred)

        print("\nTrain Acc = %0.6f" % (train_accuracy))
        print("Train Error = %0.6f" % (1. - train_accuracy))

        print("\nTest Acc = %0.6f" % (test_accuracy))
        print("Test Error = %0.6f" % (1. - test_accuracy))

        print("\n\n")
        print("Finished training !")


def evaluate_classical_model(model_fn, dataset_name, model_name=None):
    """
    Trains the base classifier (aka the attacked classifier).

    Trains only subclasses of BaseClassicalModel.

    Args:
        model_fn: A callable function that returns a subclassed BaseClassicalModel.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        model_name: Name of the model being built.

    Returns:
        (train_accuracy, test_accuracy)
    """
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = generic_utils.load_dataset(dataset_name)
    num_classes = y_train.shape[-1]

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    # construct the model on the correct device
    if model_name is not None:
        model = model_fn(num_classes, name=model_name)  # type: generic_utils.BaseClassicalModel
    else:
        model = model_fn(num_classes)  # type: generic_utils.BaseClassicalModel

    model_name = model.name
    basepath = 'weights/%s/%s/' % (dataset_name, model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + model_name + '.pkl'

    # restore the parameters that were saved
    model = model.restore(checkpoint_path)

    # Evaluate on train set once
    y_pred = model(X_train)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)

    train_accuracy = accuracy_score(y_train, y_pred)

    # Evaluate on test set once
    y_pred = model(X_test)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)

    test_accuracy = accuracy_score(y_test, y_pred)

    print("\nTrain Acc = %0.6f" % (train_accuracy))
    print("Train Error = %0.6f" % (1. - train_accuracy))

    print("\nTest Acc = %0.6f" % (test_accuracy))
    print("Test Error = %0.6f" % (1. - test_accuracy))

    return (train_accuracy, test_accuracy)
