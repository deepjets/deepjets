from deepjets.learning import train_model
from deepjets.models import get_maxout
from deepjets.utils import prepare_datasets
import sys

sig_file = sys.argv[1]
bkd_file = sys.argv[2]
dataset_name = sys.argv[3]
model_name = sys.argv[4]
n_images = 200000
test_frac = 0.5
val_frac = 0.1
batch_size = 100
epochs = 100
patience = 10
lr_init = 0.001
lr_scale_factor = 1.

print sig_file, bkd_file

h5_files = prepare_datasets(
   sig_file, bkd_file, dataset_name, n_sig=n_images, n_bkd=n_images,
   test_frac=test_frac, shuffle=True)
model = get_maxout(25**2)
train_model(model, h5_files['train'], model_name, batch_size=batch_size,
            val_frac=val_frac, epochs=epochs, patience=patience, log_to_file=True,
            read_into_ram=True)