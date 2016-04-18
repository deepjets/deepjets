import sys
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu{0}'.format(sys.argv[1]))

from deepjets.learning import test_model, train_model, cross_validate_model
from deepjets.models import get_maxout, load_model
from deepjets.utils import prepare_datasets
import numpy as np

n_images=400000
test_frac=0.5
sig_file='../data/w_events_j1p0_sj0p30_jets_zoomed_images.h5'
bkd_file='../data/qcd_events_j1p0_sj0p30_jets_zoomed_images.h5'
dataset_name='datasets/test_{0}'.format(sys.argv[1])
model_name='models/test_{0}_kfold'.format(sys.argv[1])
n_folds = 2

batch_size=100
epochs=100
val_frac=0.1
patience=10
lr_init=0.001
lr_scale_factor=1.

# Prepare datasets - slow, try to only do once.
h5_files = prepare_datasets(
    sig_file, bkd_file, dataset_name, n_sig=n_images, n_bkd=n_images,
    test_frac=test_frac, n_folds=n_folds, shuffle=True)

model = get_maxout(25**2)
dataset_file = h5_files['train']

"""
train_model(
    model, dataset_file,
    model_name=model_name,
    batch_size=batch_size, epochs=epochs,
    val_frac=val_frac, patience=patience,
    lr_init=lr_init, lr_scale_factor=lr_scale_factor,
    log_to_file=False, read_into_ram=True)#, max_jobs=n_folds)
"""

vals = cross_validate_model(
        model, dataset_file,
        model_name=model_name,
        batch_size=batch_size, epochs=epochs,
        val_frac=val_frac, patience=patience,
        lr_init=lr_init, lr_scale_factor=lr_scale_factor,
        log_to_file=True, read_into_ram=True, max_jobs=1)  # do not run CV in parallel
print np.array(vals['AUC']).mean()

#vals = test_model(model, h5_files['test'])
#print np.array(vals['AUC']).mean()
