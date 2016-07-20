from __future__ import print_function
import os
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ma
from sklearn import cross_validation
from sklearn.metrics import roc_curve


def load_images(image_h5_file, n_images=-1, shuffle_seed=1):
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
        auxvars = h5file['auxvars']
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
            auxvars = np.take(auxvars, keep, axis=0)
        else:
            images = h5file['images'][:]
            auxvars = h5file['auxvars'][:]
    return images, auxvars


def prepare_datasets(
        sig_h5_file, bkd_h5_file, dataset_name='dataset', n_sig=-1, n_bkd=-1,
        test_frac=0.1, val_frac=0., n_folds=1,
        shuffle=True, shuffle_seed=1, balance=True):
    """Prepare datasets for network training.

    Combine signal and background images; k-fold into training, validation,
    test sets. Save to files.

    Args:
        sig_h5_file, bkd_h5_file: location of h5 files containing signal,
                                  background images.
        dataset_name: base filename to use for saving datasets.
        n_sig, n_bkd: number of signal, background images to load.
        test_frac: proportion of images to save for testing.
        val_frac: proportion of images to save for validation. Leave at zero
                  unless using ROC AUC scoring.
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
    sig_images, sig_auxvars = load_images(sig_h5_file, n_sig)
    bkd_images, bkd_auxvars = load_images(bkd_h5_file, n_bkd)
    if balance:
        # crop to same size
        n = min(len(sig_images), len(bkd_images))
        sig_images = sig_images[:n]
        sig_auxvars = sig_auxvars[:n]
        bkd_images = bkd_images[:n]
        bkd_auxvars = bkd_auxvars[:n]
    n_sig = len(sig_images)
    n_bkd = len(bkd_images)
    print("collected {0} signal and {1} background images".format(n_sig, n_bkd))
    n_images = n_sig + n_bkd
    images = np.concatenate((sig_images, bkd_images))
    images = images.reshape(-1, images.shape[1] * images.shape[2])
    auxvars = np.concatenate((sig_auxvars, bkd_auxvars))
    # True classes
    classes = np.concatenate([np.repeat([[1, 0]], n_sig, axis=0),
                              np.repeat([[0, 1]], n_bkd, axis=0)])
    # Test images only
    if test_frac >= 1:
        out_file = dataset_name+'_test.h5'
        with h5py.File(out_file, 'w') as h5file:
            h5file.create_dataset('X_test', data=images)
            h5file.create_dataset('Y_test', data=classes)
            h5file.create_dataset('auxvars_test', data=auxvars)
        return {'test' : out_file}
    # Train images only
    if test_frac <= 0:
        # Shuffle to make sure validation set contains both classes
        rs = np.random.RandomState(shuffle_seed)
        train = np.arange(len(images))
        rs.shuffle(train)
        out_file = dataset_name+'_train.h5'
        with h5py.File(out_file, 'w') as h5file:
            n_val = int(val_frac * len(train))
            h5file.create_dataset('X_val', data=images[train][:n_val])
            h5file.create_dataset('Y_val', data=classes[train][:n_val])
            h5file.create_dataset(
                'auxvars_val', data=auxvars[train][:n_val])
            h5file.create_dataset('X_train', data=images[train][n_val:])
            h5file.create_dataset('Y_train', data=classes[train][n_val:])
            h5file.create_dataset(
                'auxvars_train', data=auxvars[train][n_val:])
        return {'train' : out_file}
    # Top level train-test split
    rs = cross_validation.ShuffleSplit(
        n_images, n_iter=1, test_size=test_frac, random_state=shuffle_seed)
    for trn, tst in rs:
        train, test = trn, tst
    out_file = dataset_name+'_test.h5'
    with h5py.File(out_file, 'w') as h5file:
        h5file.create_dataset('X_test', data=images[test])
        h5file.create_dataset('Y_test', data=classes[test])
        h5file.create_dataset('auxvars_test', data=auxvars[test])
    file_dict = {'test' : out_file}
    # K-fold train-val-test splits
    if n_folds > 1:
        kf = cross_validation.KFold(
            len(train), n_folds, shuffle=True, random_state=shuffle_seed)
        i = 0
        kf_files = []
        for ktrain, ktest in kf:
            # Shuffle to make sure validation set contains both classes
            np.random.shuffle(ktrain)
            out_file = dataset_name+'_train_kf{0}.h5'.format(i)
            with h5py.File(out_file, 'w') as h5file:
                h5file.create_dataset('X_test', data=images[train][ktest])
                h5file.create_dataset('Y_test', data=classes[train][ktest])
                h5file.create_dataset(
                    'auxvars_test', data=auxvars[train][ktest])
                n_val = int(val_frac * len(ktrain))
                h5file.create_dataset(
                    'X_val', data=images[train][ktrain][:n_val])
                h5file.create_dataset(
                    'Y_val', data=classes[train][ktrain][:n_val])
                h5file.create_dataset(
                    'auxvars_val', data=auxvars[train][ktrain][:n_val])
                h5file.create_dataset(
                    'X_train', data=images[train][ktrain][n_val:])
                h5file.create_dataset(
                    'Y_train', data=classes[train][ktrain][n_val:])
                h5file.create_dataset(
                    'auxvars_train', data=auxvars[train][ktrain][n_val:])
            kf_files.append(out_file)
            i += 1
        file_dict['train'] = kf_files
    else:
        # Shuffle to make sure validation set contains both classes
        rs = np.random.RandomState(shuffle_seed)
        rs.shuffle(train)
        out_file = dataset_name+'_train.h5'
        with h5py.File(out_file, 'w') as h5file:
            n_val = int(val_frac * len(train))
            h5file.create_dataset('X_val', data=images[train][:n_val])
            h5file.create_dataset('Y_val', data=classes[train][:n_val])
            h5file.create_dataset(
                'auxvars_val', data=auxvars[train][:n_val])
            h5file.create_dataset('X_train', data=images[train][n_val:])
            h5file.create_dataset('Y_train', data=classes[train][n_val:])
            h5file.create_dataset(
                'auxvars_train', data=auxvars[train][n_val:])
        file_dict['train'] = out_file
    return file_dict


def get_weights(pt, pt_min, pt_max, pt_bins):
    # Compute weights such that pT distribution is flat
    pt_hist, edges = np.histogram(
        pt, bins=np.linspace(pt_min, pt_max, pt_bins + 1))
    # Normalize
    pt_hist = np.true_divide(pt_hist, pt_hist.sum())
    image_weights = np.true_divide(
        1., np.take(pt_hist, np.searchsorted(edges, pt) - 1))
    image_weights = np.true_divide(image_weights, image_weights.mean())
    return image_weights


def make_flat_sample(image_h5_file, pt_min, pt_max, pt_bins=20):
    """ Crop and weight a dataset such that pt is within pt_min and pt_max
    and the pt distribution is approximately flat. Return the images and
    weights.
    """
    with h5py.File(image_h5_file, 'r') as h5file:
        images = h5file['images']
        auxvars = h5file['auxvars']
        jet_pt_accept = ((auxvars['pt_trimmed'] >= pt_min) &
                         (auxvars['pt_trimmed'] < pt_max))
        images = np.take(images, np.where(jet_pt_accept), axis=0)
        jet_pt = auxvars['pt_trimmed'][jet_pt_accept]
    weights = get_weights(jet_pt, pt_min, pt_max, pt_bins)
    return images, weights


class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint


def plot_jet_image(
        ax, image, vmin=1e-9, vmax=1e-2, cmap="jet", title="Intensity",
        label_axes=True, show_colorbar=True, colorbar_inside=False):
    """Display jet image.

    Args:
        ax: matplotlib axes to plot on.
        image: array representing image to plot.
        vmin, vmax: min, max intensity values to plot.
    """
    width, height = image.T.shape
    dw, dh = 1./width, 1./height
    if not (vmin is None) and not (vmax is None):
        if vmin < 0:
            norm = MidPointNorm(vmin=vmin, vmax=vmax)
            ticks = None
        else:
            norm = LogNorm(vmin=vmin, vmax=vmax)
            ticks = np.logspace(
                np.log10(vmin), np.log10(vmax),
                1 + np.log10(vmax) - np.log10(vmin))
    else:
        norm = None
        ticks = None
    p = ax.imshow(
        image.T, extent=(-(1+dw), 1+dw, -(1+dh), 1+dh), origin='low',
        interpolation='nearest', norm=norm, cmap=cmap)
    if show_colorbar:
        if colorbar_inside:
            cax = ax.figure.add_axes([0.85, 0.08, 0.03, 0.82])
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(p, cax=cax, ticks=ticks)
        cbar.set_label(title, rotation=90, fontsize=18)
        cbar.ax.tick_params(labelsize=12)
        if colorbar_inside:
            cbar.ax.yaxis.set_ticks_position('left')
    if label_axes:
        ax.set_xlabel(r'$x_1$', fontsize=18)
        ax.set_ylabel(r'$x_2$', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)
    else:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])


def plot_sig_bkd_dists(
        model, test_h5_file, batch_size=32, X_dataset='X_test',
        Y_dataset='Y_test', legend_loc=2):
    """Test model. Display signal and background distributions.

    Args:
        model: keras model to test.
        test_h5_file: name of h5 file containing test datasets.
        batch_size: integer to pass to keras.
        X_dataset: name of X_test dataset.
        Y_dataset: name of Y_test dataset.
        legend_loc: int for matplotlib legend location.
    """
    with h5py.File(test_h5_file, 'r') as h5file:
        Y_test = h5file[Y_dataset][:]
        Y_prob = model.predict_proba(
            h5file[X_dataset], batch_size=batch_size, verbose=0)
    Y_prob /= Y_prob.sum(axis=1)[:, np.newaxis]
    sig_prob = np.array([p[0] for p, y in zip(Y_prob, Y_test) if y[0] == 1])
    bkd_prob = np.array([p[0] for p, y in zip(Y_prob, Y_test) if y[0] == 0])
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    bins = np.linspace(0, 1, 50)
    ax.hist(
        sig_prob, bins=bins, histtype='stepfilled', normed=True, color='b',
        alpha=0.5, label='signal')
    ax.hist(
        bkd_prob, bins=bins, histtype='stepfilled', normed=True, color='r',
        alpha=0.5, label='background')
    ax.set_xlabel('network output', fontsize=16)
    ax.set_ylabel('frequency', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=16, loc=legend_loc)
    fig.show()


def plot_gen_dists(
        model, test_h5_files, labels, batch_size=32, X_datasets=None,
        Y_datasets=None, legend_loc=2):
    """Test model. Display signal and background distributions.

    Args:
        model: keras model to test.
        test_h5_files: name of h5 files containing test datasets.
        labels: labels for each dataset.
        batch_size: integer to pass to keras.
        X_datasets: name of X_test datasets.
        Y_datasets: name of Y_test datasets.
        legend_loc: int for matplotlib legend location.
    """
    if X_datasets is None:
        X_datasets = ['X_train'] * len(test_h5_files)
    if Y_datasets is None:
        Y_datasets = ['Y_train'] * len(test_h5_files)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    bins = np.linspace(0, 1, 50)
    for test_h5_file, label, X_dataset, Y_dataset in zip(
            test_h5_files, labels, X_datasets, Y_datasets):
        with h5py.File(test_h5_file, 'r') as h5file:
            Y_test = h5file[Y_dataset][:]
            Y_prob = model.predict_proba(
                h5file[X_dataset], batch_size=batch_size, verbose=0)
            Y_prob /= Y_prob.sum(axis=1)[:, np.newaxis]
            sig_prob = np.array([p[0] for p, y in zip(Y_prob, Y_test)
                                 if y[0] == 1])
            ax.hist(
                sig_prob, bins=bins, histtype='stepfilled', normed=True,
                alpha=0.5, label=label)
    ax.set_xlabel('network output', fontsize=16)
    ax.set_ylabel('frequency', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=16, loc=legend_loc)
    fig.show()


def likelihood_ratio(Y_true, var, sample_weight=None, nb_per_bin=1):
    """Signal to background likelihood ratio for a single variable.

    Ratio is regularised by binning such that the same number of background
    images is in each bin.

    Args:
        Y_true: array of true classes (n*2).
        var: array of variable values.
        sample_weight: array of sample weights.
        nb_per_bin: number of background samples to include in each bin.
    Returns:
        lklhd_rat: array of signal to background likelihood ratios for bins.
        bins: array of bin edges in var.
    """
    var_s = var[Y_true[:, 0] == 1]
    var_b = var[Y_true[:, 0] == 0]
    argsort_b = var_b.argsort()
    var_b = var_b[argsort_b]
    if sample_weight is not None:
        weights_s = sample_weight[Y_true[:, 0] == 1]
        weights_b = sample_weight[Y_true[:, 0] == 0]
        weights_b = weights_b[argsort_b]
    else:
        weights_s = np.ones(len(var_s))
        weights_b = np.ones(len(var_b))
    bins = np.unique([var_b[i] for i in
                      xrange(0, len(var_b), nb_per_bin)] + [var_b[-1]])
    bins[0] = var.min() - 0.1*abs(var.min())
    bins[-1] = var.max() + 0.1*abs(var.max())
    h_s, _ = np.histogram(var_s, bins, weights=weights_s)
    h_b, _ = np.histogram(var_b, bins, weights=weights_b)
    h_s /= weights_s.sum()
    h_b /= weights_b.sum()
    lklhd_rat = h_s / h_b
    return (lklhd_rat, bins)


def likelihood_ratio2d(Y_true, var1, var2, sample_weight=None, nb_per_bin=1):
    """Signal to background likelihood ratio for a pair of variables.

    Ratio is regularised by binning such that the same number of background
    images is in each bin.

    Args:
        Y_true: array of true classes (n*2).
        var1: array of variable values.
        var2: array of variable values.
        sample_weight: array of sample weights.
        nb_per_bin: number of background samples to include in each bin.
    Returns:
        lklhd_rat: list of array of signal to background likelihood ratios for
                   each set of bins.
        bins1: array of bin edges in var1.
        bins2: list of arrays of bin edges in var2.
    """
    var1_s = var1[Y_true[:, 0] == 1]
    var1_b = var1[Y_true[:, 0] == 0]
    var2_s = var2[Y_true[:, 0] == 1]
    var2_b = var2[Y_true[:, 0] == 0]
    argsort_b = var1_b.argsort()
    var1_b = var1_b[argsort_b]
    var2_b = var2_b[argsort_b]
    if sample_weight is not None:
        weights_s = sample_weight[Y_true[:, 0] == 1]
        weights_b = sample_weight[Y_true[:, 0] == 0]
        weights_b = weights_b[argsort_b]
    else:
        weights_s = np.ones(len(var1_s))
        weights_b = np.ones(len(var1_b))
    nb_per_slice = int(np.round(np.sqrt(len(var1_b) / nb_per_bin)))
    bins1 = np.unique(
        [var1_b[i] for i in xrange(0, len(var1_b), nb_per_slice)] +
        [var1_b[-1]])
    bins1[0] = var1.min() - 0.1*abs(var1.min())
    bins1[-1] = var1.max() + 0.1*abs(var1.max())
    bins2 = []
    lklhd_rat = []
    for i in xrange(len(bins1) - 1):
        var2_s_slice = var2_s[(var1_s >= bins1[i]) * (var1_s < bins1[i+1])]
        var2_b_slice = var2_b[(var1_b >= bins1[i]) * (var1_b < bins1[i+1])]
        argsort_b = var2_b_slice.argsort()
        var2_b_slice = var2_b_slice[argsort_b]
        w_s_slice = weights_s[(var1_s >= bins1[i]) * (var1_s < bins1[i+1])]
        w_b_slice = weights_b[(var1_b >= bins1[i]) * (var1_b < bins1[i+1])]
        w_b_slice = w_b_slice[argsort_b]
        bins2_slice = np.unique(
            [var2_b_slice[j] for j in
             xrange(0, len(var2_b_slice), nb_per_bin)] +
            [var2_b_slice[-1]])
        bins2_slice[0] = var2.min() - 0.1*abs(var2.min())
        bins2_slice[-1] = var2.max() + 0.1*abs(var2.max())
        bins2.append(bins2_slice)
        h_s, _ = np.histogram(var2_s_slice, bins2_slice, weights=w_s_slice)
        h_b, _ = np.histogram(var2_b_slice, bins2_slice, weights=w_b_slice)
        h_s /= weights_s.sum()
        h_b /= weights_b.sum()
        lklhd_rat.append(h_s / h_b)
    return (lklhd_rat, bins1, bins2)


def default_inv_roc_curve(Y_true, var, sample_weight=None):
    """Default ROC curve for a single variable.

    Args:
        Y_true: array of true classes (n*2).
        var: array of variable values.
        sample_weight: array of sample weights.
    Returns:
        Array of (signal efficiency, 1/[background efficiency]) pairs.
    """
    fpr, tpr, _ = roc_curve(Y_true[:, 0], var, sample_weight=sample_weight)
    res = 1./len(Y_true)
    return np.array([[tp, 1./max(fp, res)]
                     for tp,fp in zip(tpr,fpr)
                     if fp > 0.])


def lklhd_inv_roc_curve(Y_true, var, sample_weight=None, nb_per_bin=1):
    """Likelihood ratio ROC curve for a single variable.

    Args:
        Y_true: array of true classes (n*2).
        var: array of variable values.
        sample_weight: array of sample weights.
        nb_per_bin: number of background samples per bin for likelihood ratio.
    Returns:
        Array of (signal efficiency, 1/[background efficiency]) pairs.
    """
    lklhd_rat, bins = likelihood_ratio(Y_true, var, sample_weight, nb_per_bin)
    scores = lklhd_rat[np.digitize(var, bins) - 1]
    return default_inv_roc_curve(Y_true, scores, sample_weight)


def lklhd_inv_roc_curve2d(
        Y_true, var1, var2, sample_weight=None, nb_per_bin=1):
    """Likelihood ratio ROC curve for a pair of variables.

    Args:
        Y_true: array of true classes (n*2).
        var1: array of variable values.
        var2: array of variable values.
        sample_weight: array of sample weights.
        nb_per_bin: number of background samples per bin for likelihood ratio.
    Returns:
        Array of (signal efficiency, 1/[background efficiency]) pairs.
    """
    lklhd_rat_a, bins1_a, bins2_a = likelihood_ratio2d(
        Y_true, var1, var2, sample_weight, nb_per_bin)
    #lklhd_rat_b, bins2_b, bins1_b = likelihood_ratio2d(
    #    Y_true, var2, var1, sample_weight, nb_per_bin)
    scores = np.zeros(len(Y_true))
    for i in xrange(len(Y_true)):
        j1_a = np.digitize(var1[i], bins1_a)-1
        j2_a = np.digitize(var2[i], bins2_a[j1_a])-1
        #j2_b = np.digitize(var2[i], bins2_b)-1
        #j1_b = np.digitize(var1[i], bins1_b[j2_b])-1
        #scores[i] = 0.5*(lklhd_rat_a[j1_a][j2_a] + lklhd_rat_b[j2_b][j1_b])
        scores[i] = lklhd_rat_a[j1_a][j2_a]
    return default_inv_roc_curve(Y_true, scores, sample_weight)


def auxvar_roc_curve(
        test_h5_file, auxvar, Y_dataset='Y_test', use_lklhd_roc_curve=True):
    """Calculate ROC curve associated with auxvar.

    Args:
        test_h5_file: name of h5 file containing test datasets.
        auxvar: name of auxiliary variable to get curve for.
        Y_dataset: name of Y_test dataset.
    Returns:
        Array of ROC curve data.
    """
    with h5py.File(test_h5_file, 'r') as h5file:
        Y_test = h5file[Y_dataset][:]
        var = h5file['auxvars_test'][auxvar][:]
    if use_lklhd_roc_curve:
        return lklhd_inv_roc_curve(Y_test, var)
    else:
        return default_inv_roc_curve(Y_test, var)


def plot_roc_curve(roc_data, label=None, filename=None, logscale=False):
    """Display ROC curve.

    Args:
        roc_data: array containing ROC curve to plot.
        label: label to include in legend.
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot(roc_data[:, 0], roc_data[:, 1], label=label)
    ax.set_xlabel('signal efficiency', fontsize=16)
    ax.set_ylabel('1 / [backgroud efficiency]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.set_title("Receiver operating characteristic", fontsize=16)
    if logscale:
        ax.set_yscale("log", nonposy='clip')
    plt.legend(fontsize=16)
    if filename is not None:
        fig.savefig(filename)
    return fig


def plot_roc_curves(roc_data, labels, styles=None,
                    filename=None, logscale=False,
                    xlimits=None, ylimits=None,
                    title=None, show_ratio=False, ratio_label='Ratio',
                    linewidth=1, ratio_limits=(0.0,3.0)):
    """Display ROC curve.

    Args:
        roc_data: array containing list of ROC curves to plot.
        label: labels to include in legend.
        styles: list of linestyles.
    """
    from itertools import cycle
    fig = plt.figure(figsize=(12, 10))
    if show_ratio:
        rect_main = [0.1, 0.32, 0.85, 0.6]
        rect_ratio = [0.1, 0.1, 0.85, 0.2]
        ax = fig.add_axes(rect_main)
        ax.set_xticklabels([])
        ax_ratio = fig.add_axes(rect_ratio)
        ax_ratio.grid(True)
        ax_ratio.set_ylabel(ratio_label, fontsize=16)
    else:
        ax = fig.add_subplot(111)
    ax.grid(True)
    if styles is None:
        lines = ["-","--","-.",":"]
        linecycler = cycle(lines)
    else:
        linecycler = iter(styles)
    if xlimits is not None:
        xmin, xmax = xlimits
    lines = []
    for dat, label in zip(roc_data, labels):
        if xlimits is not None:
            dat = dat[(dat[:, 0] > xmin) & (dat[:, 0] < xmax)]
        lines.append(ax.plot(dat[:, 0], dat[:, 1], label=label,
                             ls=linecycler.next(), linewidth=linewidth)[0])
    if show_ratio:
        ax_ratio.set_xlabel('Signal Efficiency', fontsize=16)
    else:
        ax.set_xlabel('Signal Efficiency', fontsize=16)
    ax.set_ylabel('1 / [Backgroud Efficiency]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if logscale:
        ax.set_yscale("log", nonposy='clip')
    if xlimits is not None:
        ax.set_xlim(xlimits)
        if show_ratio:
            ax_ratio.set_xlim(xlimits)
    if ylimits is not None:
        ax.set_ylim(ylimits)
    ax.legend(fontsize=16)
    if title is not None:
        ax.set_title(title, fontsize=16)
    if show_ratio:
        # plot horizontal
        ax_ratio.plot([0, 1], [1, 1],
                      linewidth=lines[0].get_linewidth(),
                      color=lines[0].get_color(),
                      ls=lines[0].get_linestyle())
        for dat, line in zip(roc_data[1:], lines[1:]):
            # interpolate ratio
            interp = np.interp(roc_data[0][:, 0], dat[:, 0], dat[:, 1])
            ratio = interp / roc_data[0][:, 1]
            ax_ratio.plot(roc_data[0][:, 0], ratio,
                          ls=line.get_linestyle(),
                          linewidth=line.get_linewidth(),
                          color=line.get_color())
        ax_ratio.set_ylim(ratio_limits)
    if filename is not None:
        fig.savefig(filename)
    return fig


def tot_mom(jet_csts):
    """Jet momentum calculated from constituent 4-vectors."""
    E_tot  = np.sum(jet_csts['ET'] * np.cosh(jet_csts['eta']))
    px_tot = np.sum(jet_csts['ET'] * np.cos(jet_csts['phi']))
    py_tot = np.sum(jet_csts['ET'] * np.sin(jet_csts['phi']))
    pz_tot = np.sum(jet_csts['ET'] * np.sinh(jet_csts['eta']))
    return E_tot, px_tot, py_tot, pz_tot


def mass(E, px, py, pz):
    """Mass calculated from 4-vector."""
    m2 = E**2 - px**2 - py**2 - pz**2
    return np.sign(m2) * np.sqrt(abs(m2))


def jet_mass(jet_csts):
    """Jet mass calculated from constituent 4-vectors."""
    return mass(*tot_mom(jet_csts))


def pT(E, px, py, pz):
    """Transverse momentum calculated from 4-vector."""
    return (px**2 + py**2)**0.5


dphi = lambda phi1, phi2 : abs(math.fmod((math.fmod(phi1, 2*math.pi) -
                               math.fmod(phi2, 2*math.pi)) +
                               3*math.pi, 2*math.pi) - math.pi)


def dR(jet1, jet2):
    """Subjet separation calculated from subjet 4-vectors."""
    return ((jet1['eta'] - jet2['eta'])**2 +
            dphi(jet1['phi'], jet2['phi'])**2)**0.5
