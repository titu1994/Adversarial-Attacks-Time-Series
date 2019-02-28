import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from tqdm import tqdm

import utils.generic_utils as generic_utils


def targetted_mse(y_generated, y_pred, target_id, alpha):
    """
    Computes the MSE between y and the target class.

    Args:
        y_generated: The predicted labels of the generated images
            from the ATN.
        y_pred: predicted matrix of size (N, C).
            N is number of samples, and C is number of classes.
        target_id: integer id of the target class.
        alpha: scaling factor for target class activations.
            Must be greater than 1.

    Returns:
        loss value
    """
    y_pred = reranking(y_pred, target_id, alpha)
    loss = tf.losses.mean_squared_error(y_generated, y_pred, reduction=tf.losses.Reduction.NONE)

    return loss


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
    max_y = tf.reduce_max(y, axis=-1).numpy()

    if hasattr(y, 'numpy'):
        weighted_y = y.numpy()  # np.ones_like(y)
    else:
        weighted_y = np.copy(y)

    weighted_y[:, target] = alpha * max_y

    weighted_y = tf.convert_to_tensor(weighted_y)

    result = weighted_y
    result = result / tf.reduce_sum(result, axis=-1, keepdims=True)  # normalize to probability distribution

    return result


def compute_target_gradient(x, model, target):
    """
    Computes the gradient of the input image batch wrt the target output class.

    Note, this gradient is only ever computed from the <Student> model,
    and never from the <Teacher/Attacked> model when using this version.

    Args:
        x: batch of input of shape [B, T, C]
        model: classifier model
        target: integer id corresponding to the target class

    Returns:
        the output of the model and a list of gradients of shape [B, T, C]
    """
    with tf.GradientTape() as tape:
        tape.watch(x)  # need to watch the input tensor for grad wrt input
        out = model(x, training=False)  # in evaluation mode
        target_out = out[:, target]  # extract the target class outputs only

    image_grad = tape.gradient(target_out, x)  # compute the gradient

    return out, image_grad


def train_gatn(atn_model_fn, clf_model_fn, student_model_fn, dataset_name, target_class_id,
               alpha=1.5, beta=0.01, epochs=1, batchsize=128, lr=1e-3,
               atn_name=None, clf_name=None, student_name=None, device=None, evaluate=True):
    """
    Trains a Gradient Adversarial Transformation Network.

    Trains as a White-box / Black-box attack, and accepts
    either a Classical Model (which emits either discrete
    class labels or class probabilities) or a Neural Network
    under Black-box consideration which emits only discrete
    labels, as the target classifier.

    Args:
        atn_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        clf_model_fn: A callable function that returns a subclassed tf.keras Model
             or a subclass of BaseClassicalModel.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        student_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        target_class_id: Integer id of the target class. Ranged from [0, C-1]
            where C is the number of classes in the dataset.
        alpha: Weight of the reranking function used to compute loss Y.
        beta: Scaling weight of the reconstruction loss X.
        epochs: Number of training epochs.
        batchsize: Size of each batch.
        lr: Initial learning rate.
        atn_name: Name of the ATN model being built.
        clf_name: Name of the Classifier model being attacked.
        student_name: Name of the Student model used for the attack.
        device: Device to place the models on.
        evaluate: Whether to evaluate on the test set after training.
            This is only for observation, and takes significant time
            so it should be avoided unless absolutely necessary.
    """

    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    # Load the dataset
    (_, _), (X_test, y_test) = generic_utils.load_dataset(dataset_name)

    # Split test set to get adversarial train and test split.
    (X_train, y_train), (X_test, y_test) = generic_utils.split_dataset(X_test, y_test)

    num_classes = y_train.shape[-1]
    image_shape = X_train.shape[1:]

    # cleaning data
    idx = (np.argmax(y_train, axis=-1) != target_class_id)
    X_train = X_train[idx]
    y_train = y_train[idx]

    batchsize = min(batchsize, X_train.shape[0])

    num_train_batches = X_train.shape[0] // batchsize + int(X_train.shape[0] % batchsize != 0)
    num_test_batches = X_test.shape[0] // batchsize + int(X_test.shape[0] % batchsize != 0)

    # build the datasets
    train_dataset, test_dataset = generic_utils.prepare_dataset(X_train, y_train,
                                                                X_test, y_test,
                                                                batch_size=batchsize,
                                                                device=device)

    # construct the model on the correct device
    with tf.device(device):
        if clf_name is not None:
            clf_model = clf_model_fn(num_classes, name=clf_name)  # type: generic_utils.BaseClassicalModel
        else:
            clf_model = clf_model_fn(num_classes)  # type: generic_utils.BaseClassicalModel

        if student_name is not None:
            student_model = student_model_fn(num_classes, name=student_name)  # type: tf.keras.Model
        else:
            student_model = student_model_fn(num_classes)  # type: tf.keras.Model

        if atn_name is not None:
            atn_model = atn_model_fn(image_shape, name=atn_name)  # type: tf.keras.Model
        else:
            atn_model = atn_model_fn(image_shape)  # type: tf.keras.Model

    lr_schedule = tf.train.exponential_decay(lr, tf.train.get_or_create_global_step(),
                                             decay_steps=num_train_batches, decay_rate=0.99,
                                             staircase=True)

    optimizer = tf.train.AdamOptimizer(lr_schedule)

    atn_checkpoint = tf.train.Checkpoint(model=atn_model, optimizer=optimizer,
                                         global_step=tf.train.get_or_create_global_step())

    student_checkpoint = tf.train.Checkpoint(model=student_model)

    clf_model_name = clf_model.name if clf_name is None else clf_name
    basepath = 'weights/%s/%s/' % (dataset_name, clf_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + clf_model_name + '.pkl'

    # Restore the weights of the classifier
    if os.path.exists(checkpoint_path):
        clf_model = clf_model.restore(checkpoint_path)
        print("Classifier model restored !")

    # Restore student model
    student_model_name = student_model.name if student_name is None else student_name
    basepath = 'gatn_weights/%s/%s/' % (dataset_name, student_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    student_checkpoint_path = basepath + student_model_name

    student_checkpoint.restore(student_checkpoint_path)

    atn_model_name = atn_model.name if atn_name is None else atn_name
    gatn_basepath = 'gatn_weights/%s/%s/' % (dataset_name, atn_model_name + "_%d" % (target_class_id))

    if not os.path.exists(gatn_basepath):
        os.makedirs(gatn_basepath, exist_ok=True)

    atn_checkpoint_path = gatn_basepath + atn_model_name + "_%d" % (target_class_id)

    best_loss = np.inf

    print()

    # train loop
    for epoch_id in range(epochs):
        train_loss = tfe.metrics.Mean()
        train_acc = tfe.metrics.Mean()
        train_target_rate = tfe.metrics.Mean()

        with tqdm(train_dataset,
                  desc="Epoch %d / %d: " % (epoch_id + 1, epochs),
                  total=num_train_batches, unit=' samples') as iterator:

            for train_iter, (x, y) in enumerate(iterator):
                # Train the ATN

                if train_iter >= num_train_batches:
                    break

                with tf.GradientTape() as tape:
                    _, x_grad = compute_target_gradient(x, student_model, target_class_id)
                    x_adversarial = atn_model(x, x_grad, training=True)

                    y_pred = student_model(x, training=False)
                    y_pred_adversarial = student_model(x_adversarial, training=False)

                    loss_x = tf.losses.mean_squared_error(x, x_adversarial, reduction=tf.losses.Reduction.NONE)
                    loss_y = targetted_mse(y_pred_adversarial, y_pred, target_class_id, alpha)

                    loss_x = tf.reduce_sum(tf.reshape(loss_x, [loss_x.shape[0], -1]), axis=-1)
                    loss_y = tf.reduce_mean(loss_y, axis=-1)

                    loss_y = tf.cast(loss_y, tf.float32)

                    loss = beta * loss_x + loss_y

                # update model weights
                gradients = tape.gradient(loss, atn_model.variables)
                grad_vars = zip(gradients, atn_model.variables)

                optimizer.apply_gradients(grad_vars, tf.train.get_or_create_global_step())

                loss_val = tf.reduce_mean(loss)
                train_loss(loss_val)

                # Evaluate student for attacks
                acc_val, target_count = generic_utils.target_accuracy(y_pred, y_pred_adversarial, target_class_id)

                train_acc(acc_val)
                train_target_rate(target_count)

        print("\nTraining accuracy : %0.6f | Training num_adv : %0.6f" % (
            train_acc.result(), train_target_rate.result(),
        ))

        train_loss_val = train_loss.result()
        if best_loss > train_loss_val:
            print("Saving weights as training loss improved from %0.5f to %0.5f!" % (best_loss, train_loss_val))
            print()

            best_loss = train_loss_val

            atn_checkpoint.write(atn_checkpoint_path)

    if evaluate:
        test_acc = tfe.metrics.Mean()
        test_target_rate = tfe.metrics.Mean()

        with tqdm(test_dataset, desc='Evaluating',
                  total=num_test_batches, unit=' samples') as iterator:

            for test_iter, (x, y) in enumerate(iterator):

                if test_iter >= num_test_batches:
                    break

                _, x_test_grad = compute_target_gradient(x, student_model, target_class_id)
                x_test_adversarial = atn_model(x, x_test_grad, training=False)

                y_pred_adversarial = clf_model(x_test_adversarial, training=False)

                # compute and update the test target_accuracy
                acc_val, target_rate = generic_utils.target_accuracy(y, y_pred_adversarial, target_class_id)

                test_acc(acc_val)
                test_target_rate(target_rate)

        print("\nTest Acc = %0.6f | Target num_adv = %0.6f" % (test_acc.result(), test_target_rate.result()))

        print("\n\n")
        print("Finished training !")


def evaluate_gatn(atn_model_fn, clf_model_fn, student_model_fn, dataset_name, target_class_id,
                  batchsize=128, atn_name=None, clf_name=None, student_name=None, device=None):
    """
    Evaluates a Gradient Adversarial Transformation Network.

    Evaluates as a White-box / Black-box attack, and accepts
    either a Classical Model (which emits either discrete
    class labels or class probabilities) or a Neural Network
    under Black-box consideration which emits only discrete
    labels, as the target classifier.

    Args:
        atn_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        clf_model_fn: A callable function that returns a subclassed tf.keras Model
             or a subclass of BaseClassicalModel.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        student_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        target_class_id: Integer id of the target class. Ranged from [0, C-1]
            where C is the number of classes in the dataset.
        batchsize: Size of each batch.
        atn_name: Name of the ATN model being built.
        clf_name: Name of the Classifier model being attacked.
        student_name: Name of the Student model used for the attack.
        device: Device to place the models on.

    Returns:
        Does not return anything. This is only used for visual inspection.

        To obtain the scores, use the `train_scores_gatn` or
        `test_scores_gatn` functions.
    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    # Load the dataset
    (_, _), (X_test, y_test) = generic_utils.load_dataset(dataset_name)

    # Split test set to get adversarial train and test split.
    (X_train, y_train), (X_test, y_test) = generic_utils.split_dataset(X_test, y_test)

    num_classes = y_train.shape[-1]
    image_shape = X_train.shape[1:]

    # cleaning data
    # idx = (np.argmax(y_test, axis=-1) != target_class_id)
    # X_test = X_test[idx]
    # y_test = y_test[idx]

    batchsize = min(batchsize, X_train.shape[0])

    # num_train_batches = X_train.shape[0] // batchsize + int(X_train.shape[0] % batchsize != 0)
    num_test_batches = X_test.shape[0] // batchsize + int(X_test.shape[0] % batchsize != 0)

    # build the datasets
    _, test_dataset = generic_utils.prepare_dataset(X_train, y_train,
                                                    X_test, y_test,
                                                    batch_size=batchsize,
                                                    device=device)

    # construct the model on the correct device
    with tf.device(device):
        if clf_name is not None:
            clf_model = clf_model_fn(num_classes, name=clf_name)  # type: tf.keras.Model
        else:
            clf_model = clf_model_fn(num_classes)  # type: tf.keras.Model

        if student_name is not None:
            student_model = student_model_fn(num_classes, name=student_name)  # type: tf.keras.Model
        else:
            student_model = student_model_fn(num_classes)  # type: tf.keras.Model

        if atn_name is not None:
            atn_model = atn_model_fn(image_shape, name=atn_name)  # type: tf.keras.Model
        else:
            atn_model = atn_model_fn(image_shape)  # type: tf.keras.Model

    optimizer = tf.train.AdamOptimizer()

    atn_checkpoint = tf.train.Checkpoint(model=atn_model, optimizer=optimizer,
                                         global_step=tf.train.get_or_create_global_step())

    student_checkpoint = tf.train.Checkpoint(model=student_model)

    clf_model_name = clf_model.name if clf_name is None else clf_name
    basepath = 'weights/%s/%s/' % (dataset_name, clf_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + clf_model_name + '.pkl'

    # Restore the weights of the classifier
    if os.path.exists(checkpoint_path):
        clf_model = clf_model.restore(checkpoint_path)
        print("Classifier model restored !")

    atn_model_name = atn_model.name if atn_name is None else atn_name
    gatn_basepath = 'gatn_weights/%s/%s/' % (dataset_name, atn_model_name + "_%d" % (target_class_id))

    # Restore student model
    student_model_name = student_model.name if student_name is None else student_name
    basepath = 'gatn_weights/%s/%s/' % (dataset_name, student_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    student_checkpoint_path = basepath + student_model_name

    student_checkpoint.restore(student_checkpoint_path)

    if not os.path.exists(gatn_basepath):
        os.makedirs(gatn_basepath, exist_ok=True)

    atn_checkpoint_path = gatn_basepath + atn_model_name + "_%d" % (target_class_id)

    atn_checkpoint.restore(atn_checkpoint_path)

    # Restore the weights of the atn
    print()

    # train loop
    test_acc = tfe.metrics.Mean()
    test_target_rate = tfe.metrics.Mean()
    test_mse = tfe.metrics.Mean()

    batch_id = 0
    adversary_ids = []

    with tqdm(test_dataset, desc='Evaluating',
              total=num_test_batches, unit=' samples') as iterator:

        for test_iter, (x, y) in enumerate(iterator):

            if test_iter >= num_test_batches:
                break

            _, x_test_grad = compute_target_gradient(x, student_model, target_class_id)
            x_test_adversarial = atn_model(x, x_test_grad, training=False)

            y_test_pred = clf_model(x, training=False)
            y_pred_adversarial = clf_model(x_test_adversarial, training=False)

            # compute and update the test target_accuracy
            acc_val, target_rate = generic_utils.target_accuracy(y, y_pred_adversarial, target_class_id)

            x_mse = tf.losses.mean_squared_error(x, x_test_adversarial, reduction=tf.losses.Reduction.NONE)

            test_acc(acc_val)
            test_target_rate(target_rate)
            test_mse(x_mse)

            # find the adversary ids
            y_labels = tf.argmax(y, axis=-1).numpy().astype(int)
            y_pred_labels = generic_utils.checked_argmax(y_test_pred, to_numpy=True).astype(int)
            y_adv_labels = generic_utils.checked_argmax(y_pred_adversarial, to_numpy=True).astype(int)  # tf.argmax(y_pred_adversarial, axis=-1)

            pred_eq_ground = np.equal(y_labels, y_pred_labels)  # correct prediction
            pred_neq_adv_labels = np.not_equal(y_pred_labels, y_adv_labels)  # correct prediction was harmed by adversary

            found_adversary = np.logical_and(pred_eq_ground, pred_neq_adv_labels)

            not_same = np.argwhere(found_adversary)[:, 0]
            not_same = batch_id * batchsize + not_same
            batch_id += 1

            adversary_ids.extend(not_same.tolist())

    print("\n\nAdversary ids : ", adversary_ids)
    print("\n\nTest MSE : %0.5f | Test Acc = %0.6f | Target num_adv = %0.6f " % (test_mse.result(),
                                                                                 test_acc.result(),
                                                                                 test_target_rate.result(),
                                                                                 ))

    print("\n")
    print("Finished training !")


def train_scores_gatn(atn_model_fn, clf_model_fn, student_model_fn, dataset_name, target_class_id,
                      batchsize=128, atn_name=None, clf_name=None, student_name=None, device=None,
                      shuffle=True):
    """
    Evaluates a Gradient Adversarial Transformation Network and returns the metrics on
    the TRAIN SPLIT of the two splits.

    Evaluates as a White-box / Black-box attack, and accepts
    either a Classical Model (which emits either discrete
    class labels or class probabilities) or a Neural Network
    under Black-box consideration which emits only discrete
    labels, as the target classifier.

    It returns the scores of only the train set under two
    evaluation strategies :

    1) "realistic" outcome: When one is given ground truth labels to compare against.
    2) "optimistic" outcome : When we assume the classifier predictions prior to attack
                              are ground truth labels, and do not posses the real ground
                              truth labels to compare against.

    Args:
        atn_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        clf_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        student_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        target_class_id: Integer id of the target class. Ranged from [0, C-1]
            where C is the number of classes in the dataset.
        batchsize: Size of each batch.
        atn_name: Name of the ATN model being built.
        clf_name: Name of the Classifier model being attacked.
        student_name: Name of the Student model used for the attack.
        device: Device to place the models on.
        shuffle: Whether to shuffle the dataset being evaluated.

    Returns:
        (train_mse, train_acc_realistic, train_acc_optimistic, train_target_rate, adversary_ids)
    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    # Load the dataset
    (_, _), (X_test, y_test) = generic_utils.load_dataset(dataset_name)

    # Split test set to get adversarial train and test split.
    (X_train, y_train), (X_test, y_test) = generic_utils.split_dataset(X_test, y_test)

    num_classes = y_train.shape[-1]
    image_shape = X_train.shape[1:]

    # cleaning data
    # idx = (np.argmax(y_test, axis=-1) != target_class_id)
    # X_test = X_test[idx]
    # y_test = y_test[idx]

    batchsize = min(batchsize, X_train.shape[0])

    num_train_batches = X_train.shape[0] // batchsize + int(X_train.shape[0] % batchsize != 0)
    # num_test_batches = X_test.shape[0] // batchsize + int(X_test.shape[0] % batchsize != 0)

    # build the datasets
    train_dataset, _ = generic_utils.prepare_dataset(X_train, y_train,
                                                     X_test, y_test,
                                                     batch_size=batchsize,
                                                     shuffle=shuffle,
                                                     device=device)

    # construct the model on the correct device
    with tf.device(device):
        if clf_name is not None:
            clf_model = clf_model_fn(num_classes, name=clf_name)  # type: tf.keras.Model
        else:
            clf_model = clf_model_fn(num_classes)  # type: tf.keras.Model

        if student_name is not None:
            student_model = student_model_fn(num_classes, name=student_name)  # type: tf.keras.Model
        else:
            student_model = student_model_fn(num_classes)  # type: tf.keras.Model

        if atn_name is not None:
            atn_model = atn_model_fn(image_shape, name=atn_name)  # type: tf.keras.Model
        else:
            atn_model = atn_model_fn(image_shape)  # type: tf.keras.Model

    optimizer = tf.train.AdamOptimizer()

    atn_checkpoint = tf.train.Checkpoint(model=atn_model, optimizer=optimizer,
                                         global_step=tf.train.get_or_create_global_step())

    student_checkpoint = tf.train.Checkpoint(model=student_model)

    clf_model_name = clf_model.name if clf_name is None else clf_name
    basepath = 'weights/%s/%s/' % (dataset_name, clf_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + clf_model_name + '.pkl'

    # Restore the weights of the classifier
    if os.path.exists(checkpoint_path):
        clf_model = clf_model.restore(checkpoint_path)
        print("Classifier model restored !")

    atn_model_name = atn_model.name if atn_name is None else atn_name
    gatn_basepath = 'gatn_weights/%s/%s/' % (dataset_name, atn_model_name + "_%d" % (target_class_id))

    # Restore student model
    student_model_name = student_model.name if student_name is None else student_name
    basepath = 'gatn_weights/%s/%s/' % (dataset_name, student_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    student_checkpoint_path = basepath + student_model_name

    student_checkpoint.restore(student_checkpoint_path)

    if not os.path.exists(gatn_basepath):
        os.makedirs(gatn_basepath, exist_ok=True)

    atn_checkpoint_path = gatn_basepath + atn_model_name + "_%d" % (target_class_id)

    atn_checkpoint.restore(atn_checkpoint_path)

    # Restore the weights of the atn
    print()

    # train loop
    train_acc_realistic = tfe.metrics.Mean()
    train_acc_optimistic = tfe.metrics.Mean()
    train_target_rate = tfe.metrics.Mean()
    train_mse = tfe.metrics.Mean()

    batch_id = 0
    adversary_ids = []

    with tqdm(train_dataset, desc='Evaluating',
              total=num_train_batches, unit=' samples') as iterator:

        for test_iter, (x, y) in enumerate(iterator):

            if test_iter >= num_train_batches:
                break

            _, x_test_grad = compute_target_gradient(x, student_model, target_class_id)
            x_test_adversarial = atn_model(x, x_test_grad, training=False)

            y_test_pred = clf_model(x, training=False)
            y_pred_adversarial = clf_model(x_test_adversarial, training=False)

            # compute and update the test target_accuracy
            acc_val_white, target_rate = generic_utils.target_accuracy(y, y_pred_adversarial, target_class_id)
            acc_val_black, _ = generic_utils.target_accuracy(y_test_pred, y_pred_adversarial, target_class_id)

            x_mse = tf.losses.mean_squared_error(x, x_test_adversarial, reduction=tf.losses.Reduction.NONE)

            train_acc_realistic(acc_val_white)
            train_acc_optimistic(acc_val_black)
            train_target_rate(target_rate)
            train_mse(x_mse)

            # find the adversary ids
            y_labels = tf.argmax(y, axis=-1).numpy().astype(int)
            y_pred_labels = generic_utils.checked_argmax(y_test_pred, to_numpy=True).astype(int)
            y_adv_labels = generic_utils.checked_argmax(y_pred_adversarial, to_numpy=True).astype(int)  # tf.argmax(y_pred_adversarial, axis=-1)

            pred_eq_ground = np.equal(y_labels, y_pred_labels)  # correct prediction
            pred_neq_adv_labels = np.not_equal(y_pred_labels, y_adv_labels)  # correct prediction was harmed by adversary

            found_adversary = np.logical_and(pred_eq_ground, pred_neq_adv_labels)

            not_same = np.argwhere(found_adversary)[:, 0]
            not_same = batch_id * batchsize + not_same
            batch_id += 1

            adversary_ids.extend(not_same.tolist())

    return (train_mse.result().numpy(),
            train_acc_realistic.result().numpy(), train_acc_optimistic.result().numpy(),
            train_target_rate.result().numpy(), adversary_ids)


def test_scores_gatn(atn_model_fn, clf_model_fn, student_model_fn, dataset_name, target_class_id,
                     batchsize=128, atn_name=None, clf_name=None, student_name=None, device=None,
                     shuffle=True):
    """
    Evaluates a Gradient Adversarial Transformation Network and returns the metrics on
    the TEST SPLIT of the two splits.

    Evaluates as a White-box / Black-box attack, and accepts
    either a Classical Model (which emits either discrete
    class labels or class probabilities) or a Neural Network
    under Black-box consideration which emits only discrete
    labels, as the target classifier.

    It returns the scores of only the train set under two
    evaluation strategies :

    1) "realistic" outcome: When one is given ground truth labels to compare against.
    2) "optimistic" outcome : When we assume the classifier predictions prior to attack
                              are ground truth labels, and do not posses the real ground
                              truth labels to compare against.

    Args:
        atn_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        clf_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        student_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        target_class_id: Integer id of the target class. Ranged from [0, C-1]
            where C is the number of classes in the dataset.
        batchsize: Size of each batch.
        atn_name: Name of the ATN model being built.
        clf_name: Name of the Classifier model being attacked.
        student_name: Name of the Student model used for the attack.
        device: Device to place the models on.
        shuffle: Whether to shuffle the dataset being evaluated.

    Returns:
        (test_mse, test_acc_realistic, test_acc_optimistic, test_target_rate, adversary_ids)
    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    # Load the dataset
    (_, _), (X_test, y_test) = generic_utils.load_dataset(dataset_name)

    # Split test set to get adversarial train and test split.
    (X_train, y_train), (X_test, y_test) = generic_utils.split_dataset(X_test, y_test)

    num_classes = y_train.shape[-1]
    image_shape = X_train.shape[1:]

    # cleaning data
    # idx = (np.argmax(y_test, axis=-1) != target_class_id)
    # X_test = X_test[idx]
    # y_test = y_test[idx]

    batchsize = min(batchsize, X_test.shape[0])

    # num_train_batches = X_train.shape[0] // batchsize + int(X_train.shape[0] % batchsize != 0)
    num_test_batches = X_test.shape[0] // batchsize + int(X_test.shape[0] % batchsize != 0)

    # build the datasets
    _, test_dataset = generic_utils.prepare_dataset(X_train, y_train,
                                                    X_test, y_test,
                                                    batch_size=batchsize,
                                                    shuffle=shuffle,
                                                    device=device)

    # construct the model on the correct device
    with tf.device(device):
        if clf_name is not None:
            clf_model = clf_model_fn(num_classes, name=clf_name)  # type: tf.keras.Model
        else:
            clf_model = clf_model_fn(num_classes)  # type: tf.keras.Model

        if student_name is not None:
            student_model = student_model_fn(num_classes, name=student_name)  # type: tf.keras.Model
        else:
            student_model = student_model_fn(num_classes)  # type: tf.keras.Model

        if atn_name is not None:
            atn_model = atn_model_fn(image_shape, name=atn_name)  # type: tf.keras.Model
        else:
            atn_model = atn_model_fn(image_shape)  # type: tf.keras.Model

    optimizer = tf.train.AdamOptimizer()

    atn_checkpoint = tf.train.Checkpoint(model=atn_model, optimizer=optimizer,
                                         global_step=tf.train.get_or_create_global_step())

    student_checkpoint = tf.train.Checkpoint(model=student_model)

    clf_model_name = clf_model.name if clf_name is None else clf_name
    basepath = 'weights/%s/%s/' % (dataset_name, clf_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + clf_model_name + '.pkl'

    # Restore the weights of the classifier
    if os.path.exists(checkpoint_path):
        clf_model = clf_model.restore(checkpoint_path)
        print("Classifier model restored !")

    atn_model_name = atn_model.name if atn_name is None else atn_name
    gatn_basepath = 'gatn_weights/%s/%s/' % (dataset_name, atn_model_name + "_%d" % (target_class_id))

    # Restore student model
    student_model_name = student_model.name if student_name is None else student_name
    basepath = 'gatn_weights/%s/%s/' % (dataset_name, student_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    student_checkpoint_path = basepath + student_model_name

    student_checkpoint.restore(student_checkpoint_path)

    if not os.path.exists(gatn_basepath):
        os.makedirs(gatn_basepath, exist_ok=True)

    atn_checkpoint_path = gatn_basepath + atn_model_name + "_%d" % (target_class_id)

    atn_checkpoint.restore(atn_checkpoint_path)

    # Restore the weights of the atn
    print()

    # train loop
    test_acc_realistic = tfe.metrics.Mean()
    test_acc_optimistic = tfe.metrics.Mean()
    test_target_rate = tfe.metrics.Mean()
    test_mse = tfe.metrics.Mean()

    batch_id = 0
    adversary_ids = []

    with tqdm(test_dataset, desc='Evaluating',
              total=num_test_batches, unit=' samples') as iterator:

        for test_iter, (x, y) in enumerate(iterator):

            if test_iter >= num_test_batches:
                break

            _, x_test_grad = compute_target_gradient(x, student_model, target_class_id)
            x_test_adversarial = atn_model(x, x_test_grad, training=False)

            y_test_pred = clf_model(x, training=False)
            y_pred_adversarial = clf_model(x_test_adversarial, training=False)

            # compute and update the test target_accuracy
            acc_val_white, target_rate = generic_utils.target_accuracy(y, y_pred_adversarial, target_class_id)
            acc_val_black, _ = generic_utils.target_accuracy(y_test_pred, y_pred_adversarial, target_class_id)

            x_mse = tf.losses.mean_squared_error(x, x_test_adversarial, reduction=tf.losses.Reduction.NONE)

            test_acc_realistic(acc_val_white)
            test_acc_optimistic(acc_val_black)
            test_target_rate(target_rate)
            test_mse(x_mse)

            # find the adversary ids
            y_labels = tf.argmax(y, axis=-1).numpy().astype(int)
            y_pred_labels = generic_utils.checked_argmax(y_test_pred, to_numpy=True).astype(int)
            y_adv_labels = generic_utils.checked_argmax(y_pred_adversarial, to_numpy=True).astype(int)  # tf.argmax(y_pred_adversarial, axis=-1)

            pred_eq_ground = np.equal(y_labels, y_pred_labels)  # correct prediction
            pred_neq_adv_labels = np.not_equal(y_pred_labels, y_adv_labels)  # correct prediction was harmed by adversary

            found_adversary = np.logical_and(pred_eq_ground, pred_neq_adv_labels)

            not_same = np.argwhere(found_adversary)[:, 0]
            not_same = batch_id * batchsize + not_same
            batch_id += 1

            adversary_ids.extend(not_same.tolist())

    return (test_mse.result().numpy(),
            test_acc_realistic.result().numpy(), test_acc_optimistic.result().numpy(),
            test_target_rate.result().numpy(), adversary_ids)


def visualise_gatn(atn_model_fn, clf_model_fn, student_model_fn, dataset_name, target_class_id, class_id=0, sample_id=0,
                   plot_delta=False, atn_name=None, clf_name=None, student_name=None, device=None, dataset_type='train',
                   save_image=False):
    """
    Visualize the generated white-box or black-box adversarial samples.

    Args:
        atn_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        clf_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        student_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        target_class_id: Integer id of the target class. Ranged from [0, C-1]
            where C is the number of classes in the dataset.
        class_id: Integer class id or None for random sample from any class.
        sample_id: Integer sample id or None for random sample from entire dataset.
        plot_delta: Whether to plot just the adversarial sample, or both the original
            and the adversarial to visually inspect the delta between the two.
        atn_name: Name of the ATN model being built.
        clf_name: Name of the Classifier model being attacked.
        student_name: Name of the Student model used for the attack.
        device: Device to place the models on.
        dataset_type: Can be "train" or "test". Decides whether to sample from
            the GATN training or testing set.
        save_image: Bool whether to save the image to file instead of plotting it.
    """
    np.random.seed(0)

    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    if dataset_type not in ['train', 'test']:
        raise ValueError("Dataset type must be 'train' or 'test'")

    # Load the dataset
    (_, _), (X_test, y_test) = generic_utils.load_dataset(dataset_name)

    # Split test set to get adversarial train and test split.
    (X_train, y_train), (X_test, y_test) = generic_utils.split_dataset(X_test, y_test)

    num_classes = y_train.shape[-1]
    image_shape = X_train.shape[1:]

    # cleaning data
    if class_id is not None:
        assert class_id in np.unique(np.argmax(y_test, axis=-1)), "Class id must be part of the labels of the dataset !"

    # construct the model on the correct device
    with tf.device(device):
        if clf_name is not None:
            clf_model = clf_model_fn(num_classes, name=clf_name)  # type: tf.keras.Model
        else:
            clf_model = clf_model_fn(num_classes)  # type: tf.keras.Model

        if student_name is not None:
            student_model = student_model_fn(num_classes, name=student_name)  # type: tf.keras.Model
        else:
            student_model = student_model_fn(num_classes)  # type: tf.keras.Model

        if atn_name is not None:
            atn_model = atn_model_fn(image_shape, name=atn_name)  # type: tf.keras.Model
        else:
            atn_model = atn_model_fn(image_shape)  # type: tf.keras.Model

    optimizer = tf.train.AdamOptimizer()

    atn_checkpoint = tf.train.Checkpoint(model=atn_model, optimizer=optimizer,
                                         global_step=tf.train.get_or_create_global_step())

    student_checkpoint = tf.train.Checkpoint(model=student_model)

    clf_model_name = clf_model.name if clf_name is None else clf_name
    basepath = 'weights/%s/%s/' % (dataset_name, clf_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + clf_model_name + '.pkl'

    # Restore the weights of the classifier
    if os.path.exists(checkpoint_path):
        clf_model = clf_model.restore(checkpoint_path)
        print("Classifier model restored !")

    # Restore student model
    student_model_name = student_model.name if student_name is None else student_name
    basepath = 'gatn_weights/%s/%s/' % (dataset_name, student_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    student_checkpoint_path = basepath + student_model_name

    student_checkpoint.restore(student_checkpoint_path)

    atn_model_name = atn_model.name if atn_name is None else atn_name
    gatn_basepath = 'gatn_weights/%s/%s/' % (dataset_name, atn_model_name + "_%d" % (target_class_id))

    if not os.path.exists(gatn_basepath):
        os.makedirs(gatn_basepath, exist_ok=True)

    atn_checkpoint_path = gatn_basepath + atn_model_name + "_%d" % (target_class_id)

    atn_checkpoint.restore(atn_checkpoint_path)

    # Restore the weights of the atn
    print()

    sample_idx = sample_id  # np.random.randint(0, len(X_test))

    if class_id is None:

        if dataset_type == 'train':
            x = X_train[[sample_idx]]
            y = np.argmax(y_train[sample_idx])

        else:
            x = X_test[[sample_idx]]
            y = np.argmax(y_test[sample_idx])

    else:
        if dataset_type == 'train':
            class_indices = (np.argmax(y_train, axis=-1) == class_id)
            print("Number of samples of class %d = %d" % (class_id, np.sum(class_indices)))

            x = X_train[class_indices][[sample_idx]]
            y = np.argmax(y_train[class_indices][sample_idx])

        else:
            class_indices = (np.argmax(y_test, axis=-1) == class_id)
            print("Number of samples of class %d = %d" % (class_id, np.sum(class_indices)))

            x = X_test[class_indices][[sample_idx]]
            y = np.argmax(y_test[class_indices][sample_idx])

    x = tf.convert_to_tensor(x)

    _, x_test_grad = compute_target_gradient(x, student_model, target_class_id)
    x_test_adversarial = atn_model(x, x_test_grad, training=False)

    y_pred_label = clf_model(x, training=False)
    y_pred_adversarial = clf_model(x_test_adversarial, training=False)
    y_pred_adversarial_label = generic_utils.checked_argmax(y_pred_adversarial)[0]  # tf.argmax(y_pred_adversarial[0], axis=-1)

    y_pred_class = generic_utils.checked_argmax(y_pred_label, to_numpy=True)[0]  # np.argmax(y_pred_label)
    y_pred_label = np.array(y_pred_label)
    y_pred_adversarial = np.array(y_pred_adversarial)

    if y_pred_label.ndim > 1:
        y_pred_proba = y_pred_label[0, y_pred_class]
    else:
        y_pred_proba = np.nan

    if y_pred_adversarial.ndim > 1:
        y_adversarial_pred_proba = y_pred_adversarial[0, y_pred_adversarial_label]
    else:
        y_adversarial_pred_proba = np.nan

    mse_loss = tf.losses.mean_squared_error(x, x_test_adversarial)

    print("Ground truth : ", y)
    print("Real predicted probability (class = %d) : " % (y_pred_class), y_pred_proba)
    print("Adverarial predicted probability (class = %d) : " % (y_pred_adversarial_label), y_adversarial_pred_proba)
    print("Mean Squared error between X and X' : %0.6f" % (mse_loss))

    if hasattr(x_test_adversarial, 'numpy'):
        x_test_adversarial = x_test_adversarial.numpy()

    if hasattr(y_pred_adversarial_label, 'numpy'):
        y_pred_adversarial_label = y_pred_adversarial_label.numpy()

    if plot_delta:
        fig, axes = plt.subplots(1, 1, sharex=True, squeeze=True, figsize=(12, 8))

        generic_utils.plot_image_adversary(x.numpy(), y, axes, imlabel='Real X')
        generic_utils.plot_image_adversary(x_test_adversarial,
                                           'Adversarial Label : ' + str(y_pred_adversarial_label) + (
                                                ' - Real label : ' + str(y)
                                           ),
                                           axes, remove_axisgrid=False,
                                           xlabel='Timesteps', ylabel='Magnitude',
                                           imlabel='Adversarial X', legend=True)

    else:
        fig, axes = plt.subplots(1, 2, sharex=True, squeeze=True, figsize=(12, 8))

        generic_utils.plot_image_adversary(x.numpy(), y, axes[0], imlabel='Real X')
        generic_utils.plot_image_adversary(x_test_adversarial,
                                           'Adversarial Label : ' + str(y_pred_adversarial_label) + (
                                                ' - Real label : ' + str(y)
                                           ),
                                           axes[1],
                                           imlabel='Adversarial Label',
                                           xlabel='Timesteps', ylabel='Magnitude',
                                           legend=True)

    if save_image:
        if not os.path.exists('images/'):
            os.makedirs('images/')

        dataset_id = int(dataset_name[4:])
        if sample_id is None:
            sample_id = -1

        filename = 'images/blackbox-dataset-%d-sample-%d.png' % (dataset_id, sample_id)
        plt.savefig(filename)

    plt.show()


def visualise_gatn_dtw(atn_model_fn, clf_model_fn, student_model_fn, dataset_name, target_class_id, class_id=0, sample_id=0,
                       plot_delta=False, atn_name=None, clf_name=None, student_name=None, device=None,
                       dataset_type='train', save_image=False):
    """
    Visualize the generated white-box or black-box adversarial samples for
    the Dynamic Time Warping Classifier.

    Performs a comparitive study, where it plots not just the original and
    the adversarial sample, but also the sample closest to the original
    and adversarial sample in the train set of the Dynamic Time Warping
    Classifier.

    Args:
        atn_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        clf_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        student_model_fn: A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        target_class_id: Integer id of the target class. Ranged from [0, C-1]
            where C is the number of classes in the dataset.
        class_id: Integer class id or None for random sample from any class.
        sample_id: Integer sample id or None for random sample from entire dataset.
        plot_delta: Whether to plot just the adversarial sample, or both the original
            and the adversarial to visually inspect the delta between the two.
        atn_name: Name of the ATN model being built.
        clf_name: Name of the Classifier model being attacked.
        student_name: Name of the Student model used for the attack.
        device: Device to place the models on.
        dataset_type: Can be "train" or "test". Decides whether to sample from
            the GATN training or testing set.
        save_image: Bool whether to save the image to file instead of plotting it.
    """
    np.random.seed(0)

    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    if dataset_type not in ['train', 'test']:
        raise ValueError("Dataset type must be 'train' or 'test'")

    # Load the dataset
    (_, _), (X_test, y_test) = generic_utils.load_dataset(dataset_name)

    # Split test set to get adversarial train and test split.
    (X_train, y_train), (X_test, y_test) = generic_utils.split_dataset(X_test, y_test)

    num_classes = y_train.shape[-1]
    image_shape = X_train.shape[1:]

    # cleaning data
    if class_id is not None:
        assert class_id in np.unique(np.argmax(y_test, axis=-1)), "Class id must be part of the labels of the dataset !"

    # construct the model on the correct device
    with tf.device(device):
        if clf_name is not None:
            clf_model = clf_model_fn(num_classes, name=clf_name)  # type: generic_utils.BaseClassicalModel
        else:
            clf_model = clf_model_fn(num_classes)  # type: generic_utils.BaseClassicalModel

        if student_name is not None:
            student_model = student_model_fn(num_classes, name=student_name)  # type: tf.keras.Model
        else:
            student_model = student_model_fn(num_classes)  # type: tf.keras.Model

        if atn_name is not None:
            atn_model = atn_model_fn(image_shape, name=atn_name)  # type: tf.keras.Model
        else:
            atn_model = atn_model_fn(image_shape)  # type: tf.keras.Model

    optimizer = tf.train.AdamOptimizer()

    atn_checkpoint = tf.train.Checkpoint(model=atn_model, optimizer=optimizer,
                                         global_step=tf.train.get_or_create_global_step())

    student_checkpoint = tf.train.Checkpoint(model=student_model)

    clf_model_name = clf_model.name if clf_name is None else clf_name
    basepath = 'weights/%s/%s/' % (dataset_name, clf_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + clf_model_name + '.pkl'

    # Restore student model
    student_model_name = student_model.name if student_name is None else student_name
    basepath = 'gatn_weights/%s/%s/' % (dataset_name, student_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    student_checkpoint_path = basepath + student_model_name

    student_checkpoint.restore(student_checkpoint_path)

    # Restore the weights of the classifier
    if os.path.exists(checkpoint_path):
        clf_model = clf_model.restore(checkpoint_path)
        print("Classifier model restored !")

    atn_model_name = atn_model.name if atn_name is None else atn_name
    atn_basepath = 'gatn_weights/%s/%s/' % (dataset_name, atn_model_name + "_%d" % (target_class_id))

    if not os.path.exists(atn_basepath):
        os.makedirs(atn_basepath, exist_ok=True)

    atn_checkpoint_path = atn_basepath + atn_model_name + "_%d" % (target_class_id)

    atn_checkpoint.restore(atn_checkpoint_path)

    # Restore the weights of the atn
    print()

    sample_idx = sample_id  # np.random.randint(0, len(X_test))

    if class_id is None:

        if dataset_type == 'train':
            x = X_train[[sample_idx]]
            y = np.argmax(y_train[sample_idx])

        else:
            x = X_test[[sample_idx]]
            y = np.argmax(y_test[sample_idx])

    else:
        if dataset_type == 'train':
            class_indices = (np.argmax(y_train, axis=-1) == class_id)
            print("Number of samples of class %d = %d" % (class_id, np.sum(class_indices)))

            x = X_train[class_indices][[sample_idx]]
            y = np.argmax(y_train[class_indices][sample_idx])

        else:
            class_indices = (np.argmax(y_test, axis=-1) == class_id)
            print("Number of samples of class %d = %d" % (class_id, np.sum(class_indices)))

            x = X_test[class_indices][[sample_idx]]
            y = np.argmax(y_test[class_indices][sample_idx])

    # Find train sample closest to test sample x
    x_flattened = x.reshape((x.shape[0], -1))

    x_dist_matrix = clf_model.model._dist_matrix(x_flattened, clf_model.model.x)  # [1, N_train]
    x_train_min_idx = np.argmin(x_dist_matrix, axis=-1).flatten()  # [1,]

    x_train_sequence = clf_model.model.x[x_train_min_idx]
    y_train_label = generic_utils.checked_argmax(clf_model.model.y[x_train_min_idx], to_numpy=True)

    x = tf.convert_to_tensor(x)

    _, x_grad = compute_target_gradient(x, student_model, target_class_id)
    x_test_adversarial = atn_model(x, x_grad, training=False)

    # Find the train sample closest to the adversarial test sample x
    x_adv_numpy = x_test_adversarial.numpy()
    x_adv_numpy = x_adv_numpy.reshape((x_adv_numpy.shape[0], -1))

    x_adv_dist_matrix = clf_model.model._dist_matrix(x_adv_numpy, clf_model.model.x)  # [1, N_train]
    x_adv_min_idx = np.argmin(x_adv_dist_matrix, axis=-1).flatten()  # [1,]

    x_adv_train_sequence = clf_model.model.x[x_adv_min_idx]
    y_adv_train_label = generic_utils.checked_argmax(clf_model.model.y[x_adv_min_idx], to_numpy=True)

    y_pred_label = clf_model(x, training=False)
    y_pred_class = generic_utils.checked_argmax(y_pred_label, to_numpy=True)  # np.argmax(y_pred_label[0])

    y_pred_adversarial = clf_model(x_test_adversarial, training=False)
    y_pred_adversarial_label = generic_utils.checked_argmax(y_pred_adversarial, to_numpy=True)  # tf.argmax(y_pred_adversarial[0], axis=-1)

    y_pred_label = np.array(y_pred_label)
    y_pred_adversarial = np.array(y_pred_adversarial)

    if y_pred_label.ndim > 1:
        y_pred_proba = y_pred_label[0, y_pred_class]
    else:
        y_pred_proba = np.nan

    if y_pred_adversarial.ndim > 1:
        y_adversarial_pred_proba = y_pred_adversarial[0, y_pred_adversarial_label]
    else:
        y_adversarial_pred_proba = np.nan

    mse_loss = tf.losses.mean_squared_error(x, x_test_adversarial)

    print("Ground truth : ", y)
    print("Real predicted probability (class = %d) : " % (y_pred_class), y_pred_proba)
    print("Adverarial predicted probability (class = %d) : " % (y_pred_adversarial_label), y_adversarial_pred_proba)
    print("Mean Squared error between X and X' : %0.6f" % (mse_loss))

    if hasattr(x_test_adversarial, 'numpy'):
        x_test_adversarial = x_test_adversarial.numpy()

    if hasattr(y_pred_adversarial_label, 'numpy'):
        y_pred_adversarial_label = y_pred_adversarial_label.numpy()

    if plot_delta:
        fig, axes = plt.subplots(1, 1, sharex=True, squeeze=True, figsize=(12, 8))

        generic_utils.plot_image_adversary(x.numpy(), y, axes)
        generic_utils.plot_image_adversary(x_test_adversarial, y_pred_adversarial, axes)

    else:
        fig, axes = plt.subplots(2, 2, sharex=False, squeeze=True, figsize=(12, 8))

        generic_utils.plot_image_adversary(x.numpy(), '',
                                           imlabel='X Test (' + str(y.flatten()[0]) + ')',
                                           xlabel='Timesteps', ylabel='Magnitude',
                                           legend=True,
                                           ax=axes[0][0])

        generic_utils.plot_image_adversary(x_test_adversarial, '',
                                           imlabel='X Adv. Test (' + str(y_pred_adversarial_label.flatten()[0]) + ')',
                                           xlabel='Timesteps', ylabel='Magnitude',
                                           legend=True,
                                           ax=axes[0][1])

        generic_utils.plot_image_adversary(x_train_sequence, '',
                                           imlabel='Closest Real Train (' + str(y_train_label.flatten()[0]) + ')',
                                           xlabel='Timesteps', ylabel='Magnitude',
                                           legend=True, color='r', alpha=0.75,
                                           ax=axes[1][0])

        generic_utils.plot_image_adversary(x_adv_train_sequence, '',
                                           imlabel='Closest Adv. Train (' + str(y_adv_train_label.flatten()[0]) + ')',
                                           xlabel='Timesteps', ylabel='Magnitude',
                                           legend=True, color='r', alpha=0.75,
                                           ax=axes[1][1])

    if not save_image:
        plt.show()
    else:
        if not os.path.exists('images/'):
            os.makedirs('images/')

        dataset_id = int(dataset_name[4:])
        if sample_id is None:
            sample_id = -1

        filename = 'images/blackbox-dataset-%d-sample-%d-comparison.png' % (dataset_id, sample_id)
        plt.savefig(filename)
