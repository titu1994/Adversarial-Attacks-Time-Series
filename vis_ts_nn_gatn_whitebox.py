import numpy as np
import tensorflow as tf

from models.timeseries import gatn, base
from utils.ts_nn.gatn_utils import visualise_gatn, train_scores_gatn, test_scores_gatn

tf.enable_eager_execution()
np.random.seed(0)


if __name__ == '__main__':
    # Select the model function which builds the model
    # Ensure that either :
    # 1) You set the model name in the constructor of the Model
    # 2) You set the model name in the train_base function
    atn_model_fn = gatn.TSFullyConnectedGATN

    # Select the model function which builds the model
    # Ensure that either :
    # 1) You set the model name in the constructor of the Model
    # 2) You set the model name in the train_base function
    clf_model_fn = base.TSFullyConvolutionalNetwork

    # train the model on a dataset
    dataset = 0

    # Which class to select as target
    TARGET_CLASS = 0

    CLASS_ID = None
    SAMPLE_ID = 0

    DATASET_TYPE = 'train'

    # Model name (used iff not None). Defaults to Model.name if this is None.
    clf_model_name = 'gridsearch-' + clf_model_fn.__name__
    atn_model_name = 'gridsearch-' + atn_model_fn.__name__

    dataset = 'ucr/%s' % (str(dataset))

    tf.keras.backend.reset_uids()
    tf.keras.backend.clear_session()

    if DATASET_TYPE == 'train':
        mse, acc_white, acc_black, rate, ids = train_scores_gatn(atn_model_fn, clf_model_fn, dataset, TARGET_CLASS,
                                                                 atn_name=atn_model_name, clf_name=clf_model_name,
                                                                 shuffle=False)

    else:
        mse, acc_white, acc_black, rate, ids = test_scores_gatn(atn_model_fn, clf_model_fn, dataset, TARGET_CLASS,
                                                                atn_name=atn_model_name, clf_name=clf_model_name,
                                                                shuffle=False)

    print("Adversarial IDS : ", ids)

    tf.keras.backend.reset_uids()
    tf.keras.backend.clear_session()

    visualise_gatn(atn_model_fn, clf_model_fn, dataset, TARGET_CLASS, class_id=CLASS_ID, sample_id=SAMPLE_ID,
                   plot_delta=True, atn_name=atn_model_name, clf_name=clf_model_name, dataset_type=DATASET_TYPE,
                   save_image=False)
