# Adversarial Attacks on Time Series

Codebase for the paper [Adversarial Attacks on Time Series](https://arxiv.org/abs/1902.10755) that can be used to create adversarial samples using Gradient Adversarial Transformation Network to attack either Neural Networks (LeNet-5, Fully Convolutional Network) or Classical Models (1-Nearest Neighbor Dynamic Time Warping Classifier with 100% Warping Window).

The codebase can be used to execute either White-box or Black-box attacks on either NN's or Classical Models, and therefore the scripts are seperated into 4 different categories.

# Installation

Download the repository and apply `pip install -r requirements.txt` to install the required libraries. Please note that this library uses Tensorflow 1.12 with Eager Execution enabled. However, as Tensorflow comes in CPU and GPU variants, we default to the **CPU** variant.

In order to use Tensorflow with the **GPU**, please execute `pip install --upgrade tensorflow-gpu` **before** the running the requirments script above.

Once the required libraries have been installed, the data can be obtained as a zip file from here - http://www.cs.ucr.edu/~eamonn/time_series_data/

Extract that into some folder and it will give 125 different folders. Copy-paste the util script `extract_all_datasets.py` (found inside `utils`) to this folder and run it to get a single folder `_data` with all 125 datasets extracted. Cut-paste these files into the root of the project and rename it as the `data` directory.

# Training and Evaluation

There are 3 scripts each for the combination listed below : 

- White-box attack on Neural Network : `search_ts_nn_gatn_whitebox.py`, `eval_ts_nn_gatn_whitebox.py`, `vis_ts_nn_gatn_whitebox.py`.
- Black-box attack on Neural Network : `search_ts_nn_gatn_blackbox.py`, `eval_ts_nn_gatn_blackbox.py`, `vis_ts_nn_gatn_blackbox.py`.

- White-box attack on Classical Model : `search_ts_classical_gatn_whitebox.py`, `eval_ts_classical_gatn_whitebox.py`, `vis_ts_classical_gatn_whitebox.py`
- Black-box attack on Classical Model : `search_ts_classical_gatn_blackbox.py`, `eval_ts_classical_gatn_blackbox.py`, `vis_ts_classical_gatn_blackbox.py`
-----

## Tasks 
The three tasks that can be done with these scripts is as follows:

1) Use the `search_*` script to train the `GATN`, `Base` and if necessary, `Student` networks on the UCR datasets selected in the `datasets` list. You may add or remove any of the UCR datasets as you wish.

2) Evaluate on the unseen test split of the test set using the `eval_*` script. This requires you to first train the models using the `search_*` script corresponding the this `eval_*` script.

3) Visualize the adversaries on the train or test splits of the test set using the `vis_*` scripts. This requires you to first train the models using the `search_*` script corresponding the this `vis_*` script.

-----

## Common Details Among All Scripts

The codebase is designed to be highly flexible and extensible. Therefore all of these scripts require *model builders* rather than the actual *models* themselves.

In all of these scripts, you will notice 2 or 3 variables called `atn_model_fn`, `clf_model_fn` and sometimes `student_model_fn`. These refer to the model classes rather than build the model themselves. You may replace these values with other classes, or construct your own. 

**It is important to note that when building custom Models for use, they most follow the pattern exhibited by the pre-built models - constructor which accepts the name as an argument, parameters defined in the constructor, and the call method must follow the same pattern across all models belonging to that module, otherwise the results generated will be erroneous**.

## Searching for Adversaries

This is the main script, used to create the adversarial sample generator. There are many parameters that may be edited.

- `datasets` : List of dataset ids corresponding to the ids on the UCR Archive.
- `target class`: Changing the target class may improve or harm the generation of adversaries.
- `tau`: Temperature of the scaled logits prior to softmax.
- `teacher weights`: Loss weight between distillation loss and student loss.
- `alpha`: The weight of the target class inside the reranking function.
- `beta`: List of reconstruction weights. Increasing it gives fewer adversaries with reduced MSE. Increasing it gives more adversaries with heightened MSE.

The logs generated from this script contain some useful information, such as the number of adversaries generated per beta.

#### Please Note: To conserve disk space, only the LAST beta weights are stored. As it is relatively fast to train the GATN once the classifier and student are trained, you can quickly train another GATN on the specific beta you require by observing the log files.
-----

## Evaluating the Trained GATN on unseen Test Split

As described in the paper, we utilize one randomly sampled half of the test set of the UCR dataset to train the student and the GATN model, and provide scores on this dataset inside the log files during `search`. 

We can then evaluate the generation of adversarial samples on the unseen "Test" split of the test set using the `eval*` scripts.

#### Note: Since only the last weights of beta are stored, only 1 evaluation can be performed per dataset. There aren't any parameters other than `TARGET_CLASS` in these scripts. We do provide a placeholder for `beta`, which is saved into logs, but this is a cosmetic change which does not affect the actual evaluation process. 
-----

## Visualizing the Adversarial Time Series

Once the models have been trained via `search*`, we can evaluate these models using the `viz*` scripts.

We provide optional choice over which class is visualized (using `CLASS_ID`) and even which sample id is visualized (using `SAMPLE_ID`).

Once the appropriate parameters have been set, one can visualize either the train set samples or the test set samples (of the test set split) using the parameter `DATASET_TYPE`.

The following methodology is more practical when visualizing adversaries : 

- Run the viz script with the correct parameters once and `CLASS_ID = None`, `SAMPLE_ID = 0` and read the `Adversary List = [...]` printed. These are the sample ids that have been affected by adversarial attack.

- On the second run, select an id from thie `Adversary List` and set it as `SAMPLE_ID`.
-----

# Results

> Due to a large number of tables and figures, we request that the paper be referred to see the results.

In summary, we find train time adversaries for all datasets that we try on, and subsequently also show generalization properties of the GATN on completely unseen data.


# Additional Tools

Alongside the codebase to create adversarial samples, we also provide a Numba-Optimized implementation of the basic Dynamic Time Warping with 100% warping width, found inside `classical/classification/DTW.py`. 

Not only is this implementation multithreaded and LLVM optimized, thereby acheiving high speeds even for large time series or many samples in the dataset, this implementation also provides the "Soft 1-NN" transformation of the distance matrix of DTW for classification purposes.

When using DTWProbabilistic, the "Soft 1-NN" transformation is being applied to DTW.

# Citations
@article{karim_majumdar2019insights,
  title={Adversarial Attacks on Time Series},
  author={Karim, Fazle and Majumdar, Somshubra and Darabi, Houshang },
  journal={Arxiv},
 }



