from deepjets.learning import test_model, train_model, cross_validate_model
from deepjets.models import get_maxout, load_model
from deepjets.utils import prepare_datasets
import numpy as np

n_images=1000
test_frac=0.5
sig_file='/data/edawe/public/deepjets/events/pythia/weighted_masswindow_shrinkage/w_noshrink_zoom_images.h5'
bkd_file='/data/edawe/public/deepjets/events/pythia/weighted_masswindow_shrinkage/qcd_noshrink_zoom_images.h5'
dataset_name='datasets/test'
model_name='models/test_kfold'
batch_size=200000
epochs=100
val_frac=0.1
patience=10
lr_init=0.001
lr_scale_factor=1.
n_folds = 5

# Prepare datasets - slow, try to only do once.
h5_files = prepare_datasets(
    sig_file, bkd_file, dataset_name, n_sig=n_images, n_bkd=n_images,
    test_frac=test_frac, n_folds=n_folds, shuffle=True)



def train_one_point(args):
    # Setup and train model.
    result = np.empty(args.shape[0], dtype=np.float)
    for i, row in enumerate(args):
        learning_rate, batch_size = row
        model = get_maxout(25**2)
        dataset_file = h5_files['train']
        vals = cross_validate_model(
            model, dataset_file,
            model_name=model_name + "_lr{0}_bs{1}".format(learning_rate, batch_size),
            batch_size=batch_size, epochs=epochs,
            val_frac=val_frac, patience=patience,
            lr_init=learning_rate, lr_scale_factor=lr_scale_factor,
            log_to_file=True, read_into_ram=True, max_jobs=n_folds)
        result[i] = - np.array(vals['AUC']).mean()
    return result


bounds = [(0.0001, 0.001), (50, 100)]

import GPyOpt
from numpy.random import seed
seed(12345)


BO_demo_1d = GPyOpt.methods.BayesianOptimization(f=train_one_point,   # function to optimize
                                                    bounds=bounds)          # box-constrains of the problem

BO_demo_1d.run_optimization(max_iter,                                   # evaluation budget
                                    eps=10e-6)                              # stop criterion

