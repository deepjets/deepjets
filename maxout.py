from deepjets import learning, models, utils

n_images = 200000
n_folds = 1
test_frac = 0.5
val_frac = 0.2
sig_file = 'w_images.h5'
bkd_file = 'qcd_images.h5'
dataset_name = 'w_noshrink'
model_name = 'maxout_w_noshrink'

h5_files = utils.prepare_datasets(
    sig_file, bkd_file, dataset_name, n_sig=n_images, n_bkd=n_images, test_frac=test_frac,
    val_frac=val_frac, n_folds=n_folds, auxvars=['weights'], shuffle=True, shuffle_seed=1)
model = models.get_maxout(25**2)
learning.train_model(model, h5_files['train'], model_name, log_to_file=True)