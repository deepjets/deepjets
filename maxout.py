from deepjets.learning import test_model, train_model
from deepjets.models import get_maxout, load_model
from deepjets.utils import prepare_datasets

def full_sequence_maxout(
        n_images=1000, test_frac=0.5,
        sig_file='/data/edawe/public/deepjets/events/pythia/weighted_masswindow_shrinkage/w_noshrink_zoom_images.h5',
        bkd_file='/data/edawe/public/deepjets/events/pythia/weighted_masswindow_shrinkage/qcd_noshrink_zoom_images.h5',
        dataset_name='datasets/test', model_name='models/test',
        batch_size=100, epochs=1, val_frac=0.1, patience=10, lr_init=0.001, lr_scale_factor=1.):
    """Prepares datasets, trains and tests model."""
    h5_files = prepare_datasets(
        sig_file, bkd_file, dataset_name, n_sig=n_images, n_bkd=n_images,
        test_frac=test_frac, shuffle=True)
    model = get_maxout(25**2)
    dataset_file = h5_files['train']
    train_model(
        model, dataset_file, model_name=model_name, batch_size=batch_size, epochs=epochs,
        val_frac=val_frac, patience=patience, lr_init=lr_init, lr_scale_factor=lr_scale_factor,
        log_to_file=True, read_into_ram=True)
    model = load_model(model_name)
    test_results = test_model(
            model, h5_files['test'], model_name=model_name, log_to_file=True)
    return test_results

print full_sequence_maxout()