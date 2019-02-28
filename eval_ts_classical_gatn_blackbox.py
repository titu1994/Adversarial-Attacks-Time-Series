import os

import tensorflow as tf

from models.timeseries import gatn, classical, classical_distilled
from utils.generic_utils import enable_printing, disable_printing
from utils.ts_classical.gatn_utils import test_scores_gatn

tf.enable_eager_execution()


if __name__ == '__main__':

    PRINT_OUTPUTS = True

    # Select the model function which builds the model
    # Ensure that either :
    # 1) You set the model name in the constructor of the Model
    # 2) You set the model name in the train_base function
    atn_model_fn = gatn.TSFullyConnectedGATN

    # Select the model function which builds the model
    # Ensure that either :
    # 1) You set the model name in the constructor of the Model
    # 2) You set the model name in the train_base function
    clf_model_fn = classical.DTW

    # Select the model function which builds the model
    # Ensure that either :
    # 1) You set the model name in the constructor of the Model
    # 2) You set the model name in the train_base function
    student_model_fn = classical_distilled.TSDistilledLeNet5

    # train the model on a dataset
    datasets = [
                5, 7, 8, 18, 19, 20, 21, 28, 29, 36, 37, 39, 40, 47, 48, 49, 53, 54, 63, 64, 65,
                72, 73, 79, 86, 87, 88, 92, 93, 94, 95, 96, 98, 99, 100, 104, 105, 115, 116, 117,
                118, 125
                ]

    # Which class to select as target
    TARGET_CLASS = 0

    # Hyper parameters for searching
    TEACHER_WEIGHT = 1.0
    TAU = 10.

    ALPHA = 1.5
    BETAS = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]

    log_path = 'logs/'
    log_name = 'gatn_classical_blackbox_results_test.csv'

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_path = log_path + log_name

    template = '%d,%0.5f,%0.6f,%0.6f,%0.6f,%0.6f,%d\n'

    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('dataset_id,beta,train_mse,train_acc_realistic,train_acc_optimistic,train_target_rate,num_adversaries\n')
            f.flush()

    SUCCESS = []
    ERRORS = []

    if not PRINT_OUTPUTS:
        disable_printing()

    for dataset_id in datasets:
        for beta in BETAS:

            f = open(log_path, 'a+')

            # Model name (used iff not None). Defaults to Model.name if this is None.
            clf_model_name = 'gridsearch-blackbox-' + clf_model_fn.__name__
            atn_model_name = 'gridsearch-blackbox-' + atn_model_fn.__name__
            student_model_name = 'gridsearch-blackbox-' + student_model_fn.__name__

            dataset = 'ucr/%s' % (str(dataset_id))

            # checks if base classifier is available or not
            basepath = 'weights/%s/%s/' % (dataset, clf_model_name)
            # checks if base distilled classifier is available or not
            student_basepath = 'gatn_weights/%s/%s/' % (dataset, student_model_name)

            try:
                mse, acc_realistic, acc_optimistic, rate, ids = test_scores_gatn(atn_model_fn, clf_model_fn,
                                                                                 student_model_fn,
                                                                                 dataset, TARGET_CLASS,
                                                                                 atn_name=atn_model_name,
                                                                                 clf_name=clf_model_name,
                                                                                 student_name=student_model_name)

                # enable_printing()
                print("Finished evaluating dataset %s with beta = %0.6f" % (dataset, beta))
                # disable_printing()

                update = template % (dataset_id, beta, mse, acc_realistic, acc_optimistic, rate, len(ids))

                f.write(update)
                SUCCESS.append(update)

            except Exception as e:
                print(e.with_traceback(None))

                tag = template % (dataset_id, beta, -1, -1, -1, -1, -1)
                ERRORS.append(tag)

            f.flush()
            f.close()

            print()

    if not PRINT_OUTPUTS:
        enable_printing()

    print("\nAll searches complete !")

    print("\n", "*" * 20, "SUCCESSES", "*" * 20)
    for success in SUCCESS:
        print(success, end='', flush=True)

    print("\n", "*" * 20, "ERRORS", "*" * 20)
    for err in ERRORS:
        print(err, end='', flush=True)
