import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.contrib.eager.python import tfe
from tqdm import tqdm

import utils.generic_utils as generic_utils


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


def train_distilled_base(student_model_fn, clf_model_fn, dataset_name,
                         tau=1., teacher_weight=0.9, epochs=1, batchsize=128, lr=1e-3,
                         student_name=None, clf_name=None, device=None):
    """
    Performs Model Distillation from the Base / Attacked model
    onto the Student model.

    The Base model may be either a neural
    network which emits class probabilities or class labels.
    Likewise, the Base model can also be a descendent of the
    BaseClassicalModel which emits class probabilities or class
    labels.

    The Student model *must* be a Neural Network.

    Args:
        student_model_fn:  A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        clf_model_fn:  A callable function that returns a subclassed tf.keras Model
             or a subclass of BaseClassicalModel.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        tau: Temperature of the scaled-softmax operation. With tau equal
             to 1, acts as ordinary softmax. With larger tau, the
        teacher_weight: Float value. Scales the logits prior to softmax.
             only from the Teacher model and neglect the crossentropy loss.
             If 0, the student will learn only from the cross entropy
             loss and neglect the teacher's predictions. Any value in
             between 0 and 1 will allow weighing between the distillation
             and student losses.
        epochs: Number of training epochs.
        batchsize: Size of each batch.
        lr: Initial learning rate.
        student_name: Name of the Student model used for the attack.
        clf_name: Name of the Classifier model being attacked.
        device: Device to place the models on.
    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    # Load the dataset
    (_, _), (X_test, y_test) = generic_utils.load_dataset(dataset_name)

    (X_train, y_train), (X_test, y_test) = generic_utils.split_dataset(X_test, y_test)

    num_classes = y_train.shape[-1]
    # image_shape = X_train.shape[1:]

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

    lr_schedule = tf.train.exponential_decay(lr, tf.train.get_or_create_global_step(),
                                             decay_steps=num_train_batches, decay_rate=0.99,
                                             staircase=True)

    optimizer = tf.train.AdamOptimizer(lr_schedule)

    student_checkpoint = tf.train.Checkpoint(model=student_model, optimizer=optimizer,
                                             global_step=tf.train.get_or_create_global_step())

    clf_model_name = clf_model.name if clf_name is None else clf_name
    basepath = 'weights/%s/%s/' % (dataset_name, clf_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + clf_model_name + '.pkl'

    # Restore the weights of the classifier
    if os.path.exists(checkpoint_path):
        clf_model = clf_model.restore(checkpoint_path)
        print("Classifier model restored !")

    student_model_name = student_model.name if student_name is None else student_name
    gatn_basepath = 'gatn_weights/%s/%s/' % (dataset_name, student_model_name)

    if not os.path.exists(gatn_basepath):
        os.makedirs(gatn_basepath, exist_ok=True)

    student_checkpoint_path = gatn_basepath + student_model_name

    best_loss = np.inf

    print()

    # Parameters for Model Distillation
    assert teacher_weight >= 0. and teacher_weight <= 1.

    student_weight = 1. - teacher_weight

    # train loop
    for epoch_id in range(epochs):
        train_acc = tfe.metrics.Mean()
        train_loss = tfe.metrics.Mean()

        with tqdm(train_dataset,
                  desc="Epoch %d / %d: " % (epoch_id + 1, epochs),
                  total=num_train_batches, unit=' samples') as iterator:

            for train_iter, (x, y) in enumerate(iterator):
                # Train the ATN

                if train_iter >= num_train_batches:
                    break

                with tf.GradientTape() as tape:
                    # Obtain input gradients based on substitute student model which is differentiable
                    y_pred_student = student_model(x, training=True)  # logits
                    y_pred_teacher = clf_model(x, training=False)  # probabilities

                    y_pred_student_scaled = generic_utils.rescaled_softmax(y_pred_student, num_classes, tau)  # scaled softmax
                    y_pred_teacher_scaled = generic_utils.rescaled_softmax(y_pred_teacher, num_classes, tau)  # either scales or onehots

                    y_pred_student = tf.nn.softmax(y_pred_student, axis=-1)  # unscaled softmax

                    y = tf.cast(y, tf.float32)
                    y_pred_student = tf.cast(y_pred_student, tf.float32)

                    student_loss = tf.keras.losses.categorical_crossentropy(y, y_pred_student)
                    teacher_loss = tf.keras.losses.categorical_crossentropy(y_pred_teacher_scaled,
                                                                            y_pred_student_scaled)

                    loss = student_weight * student_loss + teacher_weight * teacher_loss
                    # Loss is a vector of size (N,). Use it directly.
                    loss = tf.cast(loss, tf.float32)

                # update model weights
                gradients = tape.gradient(loss, student_model.variables)
                grad_vars = zip(gradients, student_model.variables)

                optimizer.apply_gradients(grad_vars, tf.train.get_or_create_global_step())

                loss_val = tf.reduce_mean(loss)
                acc_val = tf.keras.metrics.categorical_accuracy(y, y_pred_student)

                train_loss(loss_val)
                train_acc(acc_val)

        print("Train accuracy = %0.6f\n" % train_acc.result())

        train_loss_val = train_loss.result()
        if best_loss > train_loss_val:
            print("Saving weights as training loss improved from %0.5f to %0.5f!" % (best_loss, train_loss_val))
            print()

            best_loss = train_loss_val

            student_checkpoint.write(student_checkpoint_path)

    test_acc = tfe.metrics.Mean()

    # Restore the weights before predicting
    student_checkpoint.restore(student_checkpoint_path)

    with tqdm(test_dataset, desc='Evaluating',
              total=num_test_batches, unit=' samples') as iterator:

        for x, y in iterator:
            y_pred_student = student_model(x, training=False)

            # compute and update the test target_accuracy
            acc_val = tf.keras.metrics.categorical_accuracy(y, y_pred_student)

            test_acc(acc_val)

    print("\nStudent Test Acc = %0.6f" % (test_acc.result()))

    print("\n\n")
    print("Finished training !")


def evaluate_distilled_base(student_model_fn, clf_model_fn, dataset_name,
                            batchsize=128, student_name=None, clf_name=None, device=None):
    """
    Performs Model Distillation from the Base / Attacked model
    onto the Student model.

    The Base model may be either a neural
    network which emits class probabilities or class labels.
    Likewise, the Base model can also be a descendent of the
    BaseClassicalModel which emits class probabilities or class
    labels.

    The Student model *must* be a Neural Network.

    Args:
        student_model_fn:  A callable function that returns a subclassed tf.keras Model.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        clf_model_fn:  A callable function that returns a subclassed tf.keras Model
             or a subclass of BaseClassicalModel.
             It can access the following args passed to it:
                - name: The model name, if a name is provided.
        dataset_name: Name of the dataset as a string.
        batchsize: Size of each batch.
        student_name: Name of the Student model used for the attack.
        clf_name: Name of the Classifier model being attacked.
        device: Device to place the models on.

    Returns:
        Does not return anything. This is only used for visual inspection.
    """
    if device is None:
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

    # Load the dataset
    (_, _), (X_test, y_test) = generic_utils.load_dataset(dataset_name)

    (X_train, y_train), (X_test, y_test) = generic_utils.split_dataset(X_test, y_test)

    num_classes = y_train.shape[-1]
    # image_shape = X_train.shape[1:]

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
            clf_model = clf_model_fn(num_classes, name=clf_name)  # type: generic_utils.BaseClassicalModel
        else:
            clf_model = clf_model_fn(num_classes)  # type: generic_utils.BaseClassicalModel

        if student_name is not None:
            student_model = student_model_fn(num_classes, name=student_name)  # type: tf.keras.Model
        else:
            student_model = student_model_fn(num_classes)  # type: tf.keras.Model

    optimizer = tf.train.AdamOptimizer()

    student_checkpoint = tf.train.Checkpoint(model=student_model, optimizer=optimizer,
                                             global_step=tf.train.get_or_create_global_step())

    clf_model_name = clf_model.name if clf_name is None else clf_name
    basepath = 'weights/%s/%s/' % (dataset_name, clf_model_name)

    if not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    checkpoint_path = basepath + clf_model_name + '.pkl'

    # Restore the weights of the classifier
    if os.path.exists(checkpoint_path):
        clf_model = clf_model.restore(checkpoint_path)
        print("Classifier model restored !")

    student_model_name = student_model.name if student_name is None else student_name
    gatn_basepath = 'gatn_weights/%s/%s/' % (dataset_name, student_model_name)

    if not os.path.exists(gatn_basepath):
        os.makedirs(gatn_basepath, exist_ok=True)

    student_checkpoint_path = gatn_basepath + student_model_name

    student_checkpoint.restore(student_checkpoint_path)

    print()

    teacher_test_acc = tfe.metrics.Mean()
    test_acc = tfe.metrics.Mean()

    with tqdm(test_dataset, desc='Evaluating',
              total=num_test_batches, unit=' samples') as iterator:

        for x, y in iterator:
            y_pred_teacher = clf_model(x, training=False)
            y_pred_student = student_model(x, training=False)

            # compute and update the test target_accuracy
            y_pred_teacher = generic_utils.checked_argmax(y_pred_teacher, to_numpy=True)
            teacher_acc_val = accuracy_score(tf.argmax(y, axis=-1).numpy(), y_pred_teacher)
            acc_val = tf.keras.metrics.categorical_accuracy(y, y_pred_student)

            teacher_test_acc(teacher_acc_val)
            test_acc(acc_val)

    print("\nTeacher Test Acc = %0.6f | Student Test Acc = %0.6f" % (teacher_test_acc.result(),
                                                                     test_acc.result()))

    print("\n\n")
    print("Finished training !")
