from __future__ import print_function
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import cross_validation


def load_images(image_h5_file, n_images=-1, auxvars=[], shuffle_seed=1):
    """Load images and auxiliary data from h5 file.
    
    Args:
        image_h5_file: location of h5 file containing images.
        n_images: number of images to load, -1 loads all.
        auxvars: list of auxvar field names to load.
    Returns:
        images: array of image arrays.
        aux_data: dict of auxvar arrays.
    TODO: add support for multiple classes.
    """
    with h5py.File(image_h5_file, 'r') as h5file:
        images = h5file['images']
        auxvars_data = h5file['auxvars']
        if n_images < 0:
            n_images = len(images)
        elif n_images > len(images):
            print("Cannot load {0} images. Only {1} images in {2}".format(
                n_images, len(images), image_h5_file))
            n_images = len(images)
        if n_images < len(images):
            rs = cross_validation.ShuffleSplit(
                len(images), n_iter=1, test_size=n_images,
                random_state=shuffle_seed)
            for train, test in rs:
                keep = test
            images = np.take(images, keep, axis=0)
            auxvars_data = np.take(auxvars_data, keep, axis=0)
            aux_data = {var : auxvars_data[var] for var in auxvars}
        else:
            images = h5file['images'][:]
            aux_data = {var : auxvars_data[var][:] for var in auxvars}
        return (images, aux_data)


def prepare_datasets(
        sig_h5_file, bkd_h5_file, dataset_name='dataset', n_sig=-1, n_bkd=-1,
        test_frac=0.1, val_frac=0.1, n_folds=2, auxvars=[], shuffle=True,
        shuffle_seed=1):
    """Prepare datasets for network training.
    
    Combine signal and background images; k-fold into training, validation,
    test sets. Save to files.
    
    Args:
        sig_h5_file, bkd_h5_file: location of h5 files containing signal,
                                  background images.
        dataset_name: base filename to use for saving datasets.
        n_sig, n_bkd: number of signal, background images to load.
        test_frac, val_frac: proportion of images to save for testing,
                             validation
        n_folds: number of k-folds.
        auxvars: list of auxvar field names to load.
        shuffle: if True shuffle images before k-folding.
        shuffle_seed: seed for shuffling.
    Returns:
        file_dict: dict containing list of filenames containing train, test
                   datasets.
    TODO: add support for multiple classes.
    """
    # Load images
    sig_images, sig_aux_data = load_images(sig_h5_file, n_sig, auxvars)
    bkd_images, bkd_aux_data = load_images(bkd_h5_file, n_bkd, auxvars)
    n_sig = len(sig_images)
    n_bkd = len(bkd_images)
    n_images = n_sig + n_bkd
    images = np.concatenate((sig_images, bkd_images))
    images = images.reshape(-1, images.shape[1] * images.shape[2])
    aux_data = {var : np.concatenate((sig_aux_data[var], bkd_aux_data[var]))
                for var in auxvars}
    # True classes
    classes = np.concatenate([np.repeat([[1, 0]], n_sig, axis=0),
                              np.repeat([[0, 1]], n_bkd, axis=0)])
    
    # Top level train-test split
    rs = cross_validation.ShuffleSplit(
        n_images, n_iter=1, test_size=test_frac, random_state=shuffle_seed)
    for trn, tst in rs:
        train, test = trn, tst
    out_file = dataset_name+'_test.h5'
    with h5py.File(out_file, 'w') as h5file:
        h5file.create_dataset('X_test', data=images[test])
        h5file.create_dataset('Y_test', data=classes[test])
        for var in auxvars:
            h5file.create_dataset(var+'_test', data=aux_data[var][test])
    file_dict = {'test' : out_file}
    
    # K-fold train-val-test splits
    if n_folds > 1:
        kf = cross_validation.KFold(
            len(train), n_folds, shuffle=True, random_state=shuffle_seed)
        i = 0
        kf_files = []
        for ktrain, ktest in kf:
            out_file = dataset_name+'_train_kf{0}.h5'.format(i)
            # Shuffle to make sure validation set contains both classes
            np.random.shuffle(ktrain)
            with h5py.File(out_file, 'w') as h5file:
                h5file.create_dataset('X_test', data=images[train][ktest])
                h5file.create_dataset('Y_test', data=classes[train][ktest])
                for var in auxvars:
                    h5file.create_dataset(
                        var+'_test', data=aux_data[var][train][ktest])
                n_val = int(val_frac * len(ktrain))
                h5file.create_dataset(
                    'X_val', data=images[train][ktrain][:n_val])
                h5file.create_dataset(
                    'Y_val', data=classes[train][ktrain][:n_val])
                for var in auxvars:
                    h5file.create_dataset(
                        var+'_val', data=aux_data[var][train][ktrain][:n_val])
                h5file.create_dataset(
                    'X_train', data=images[train][ktrain][n_val:])
                h5file.create_dataset(
                    'Y_train', data=classes[train][ktrain][n_val:])
                for var in auxvars:
                    h5file.create_dataset(
                        var+'_train',
                        data=aux_data[var][train][ktrain][n_val:])
            kf_files.append(out_file)
            i += 1
        file_dict['train'] = kf_files
    else:
        out_file = dataset_name+'_train.h5'
        # Shuffle to make sure validation set contains both classes
        np.random.shuffle(train)
        with h5py.File(out_file, 'w') as h5file:
            n_val = int(val_frac * len(train))
            h5file.create_dataset('X_val', data=images[train][:n_val])
            h5file.create_dataset('Y_val', data=classes[train][:n_val])
            for var in auxvars:
                h5file.create_dataset(
                    var+'_val', data=aux_data[var][train][:n_val])
            h5file.create_dataset('X_train', data=images[train][n_val:])
            h5file.create_dataset('Y_train', data=classes[train][n_val:])
            for var in auxvars:
                h5file.create_dataset(
                    var+'_train', data=aux_data[var][train][n_val:])
        file_dict['train'] = out_file
    return file_dict
    

def plot_jet_image(ax, image, vmin=1e-9, vmax=1e-2):
    """Display jet image.
    
    Args:
        ax: matplotlib axes to plot on.
        image: array representing image to plot.
        vmin, vmax: min, max intensity values to plot.
    """
    width, height = image.T.shape
    dw, dh = 1./width, 1./height
    p = ax.imshow(
        image.T, extent=(-(1+dw), 1+dw, -(1+dh), 1+dh), origin='low',
        interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(
        p, cax=cax,
        ticks=np.logspace(
            np.log10(vmin), np.log10(vmax), 1+np.log10(vmax)-np.log10(vmin)))
    cbar.set_label(r'Intensity', rotation=90, fontsize=18)
    cbar.ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$x_1$', fontsize=18)
    ax.set_ylabel(r'$x_2$', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)


def plot_roc_curve(roc_data, label=None):
    """Display ROC curve.
    
    Args:
        roc_data: array containing ROC curve to plot.
        label: label to include in legend.
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    handle, = ax.plot(roc_data[:, 0], roc_data[:, 1], label=label)
    ax.set_xlabel("signal efficiency", fontsize=16)
    ax.set_ylabel("1 / [backgroud efficiency]", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.set_title("Receiver operating characteristic", fontsize=16)
    plt.legend(handles=[handle], fontsize=16)
    fig.show()


def plot_roc_curves(roc_data, labels):
    """Display ROC curve.
    
    Args:
        roc_data: array containing list of ROC curves to plot.
        label: labels to include in legend.
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    handles = []
    for dat, label in zip(roc_data, labels):
        handle, = ax.plot(dat[:, 0], dat[:, 1], label=label)
        handles.append(handle)
    ax.set_xlabel("signal efficiency", fontsize=16)
    ax.set_ylabel("1 / [backgroud efficiency]", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.set_title("Receiver operating characteristic", fontsize=16)
    plt.legend(handles=handles, fontsize=16)
    fig.show()


def tot_mom(jet_csts):
    E_tot  = np.sum(jet_csts['ET'] * np.cosh(jet_csts['eta']))
    px_tot = np.sum(jet_csts['ET'] * np.cos(jet_csts['phi']))
    py_tot = np.sum(jet_csts['ET'] * np.sin(jet_csts['phi']))
    pz_tot = np.sum(jet_csts['ET'] * np.sinh(jet_csts['eta']))
    return E_tot, px_tot, py_tot, pz_tot


def mass(E, px, py, pz):
    m2 = E**2 - px**2 - py**2 - pz**2
    return np.sign(m2) * np.sqrt(abs(m2))


def jet_mass(jet_csts):
    """Returns jet mass calculated from constituent 4-vectors."""
    return mass(*tot_mom(jet_csts))


def pT(E, px, py, pz):
    return (px**2 + py**2)**0.5


dphi = lambda phi1, phi2 : abs(math.fmod((math.fmod(phi1, 2*math.pi) -
                               math.fmod(phi2, 2*math.pi)) +
                               3*math.pi, 2*math.pi) - math.pi)


def dR(jet1, jet2):
    return ((jet1['eta'] - jet2['eta'])**2 +
            dphi(jet1['phi'], jet2['phi'])**2)**0.5
