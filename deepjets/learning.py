from __future__ import print_function
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
from .models import load_model, save_model
from .utils import load_images
from multiprocessing import Pool, cpu_count
from sklearn import cross_validation
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import auc, roc_curve


def prepare_datasets(
        sig_h5_file, bkd_h5_file, dataset_name='dataset', n_sig=-1, n_bkd=-1,
        test_frac=0.1, val_frac=0.1, n_folds=2, auxvars=[], shuffle=True,
        shuffle_seed=1):
    """Combine signal, background images; k-fold into training, validation,
    test sets.
    
    Returns dict with names of 'test' file and 'train' files for k-folds.
    TODO: test support for additional fields.
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


def train_model(
        model, train_h5_file, model_name='model', batch_size=32, epochs=100,
        patience=10, verbose=2, log_to_file=False, read_into_RAM=False):
    """Train model using datasets in train_h5_file. Save model with best AUC.
    
    Passes datasets to Keras directly from train_h5_file each epoch unless
    read_into_RAM=True.
    """
    save_model(model, model_name)
    if log_to_file:
        log_file = open(model_name+'_log.txt', 'w')
    else:
        log_file = sys.stdout
    epoch = 0
    best_auc = 0.
    stop_cdn = 0
    stuck_cdn = 0
    h5file = h5py.File(train_h5_file, 'r')
    if verbose >= 1:
        print("Training on {0} samples, validating on {1} samples.".format(
              len(h5file['X_train']), len(h5file['X_val'])), file=log_file)
        print("Datasets from {0}.".format(train_h5_file), file=log_file)
        sys.stdout.flush()
    if read_into_RAM:
        shuffle = False
        X_train = h5file['X_train'][:]
        Y_train = h5file['Y_train'][:]
        X_val = h5file['X_val'][:]
    else:
        shuffle = 'batch'
        X_train = h5file['X_train']
        Y_train = h5file['Y_train']
        X_val = h5file['X_val']
    try:
        sample_weights = h5file['weights_train'][:]
    except KeyError:
        sample_weights = None
    Y_val = h5file['Y_val']
    while epoch < epochs:
        # Fitting and validation
        model.fit(
            X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=0,
            sample_weight=sample_weights, shuffle=shuffle)
        Y_prob = model.predict_proba(
            X_val, batch_size=batch_size, verbose=0)
        Y_prob /= Y_prob.sum(axis=1)[:, np.newaxis]
        # Calculate AUC for custom early stopping
        fpr, tpr, _ = roc_curve(Y_val[:, 0], Y_prob[:, 0])
        res = 1. / len(Y_val)
        inv_curve = np.array(
            [[tp, 1. / max(fp, res)]
            for tp,fp in zip(tpr,fpr) if (0.2 <= tp <= 0.8 and fp > 0.)])
        try:
            current_auc = auc(inv_curve[:, 0], inv_curve[:, 1])
        except IndexError:
            current_auc = -1
            stuck_cdn += 1
        if current_auc > best_auc:
            best_auc = current_auc
            stop_cdn = 0
            save_model(model, model_name)
        else:
            stop_cdn += 1
        if log_to_file:
            print("Epoch {0}/{1}: epochs w/o increase = {2}, AUC = {3}".format(
                epoch+1, epochs, stop_cdn, current_auc), file=log_file)
        elif verbose >= 2:
            print("\r", end='')
            print("Epoch {0}/{1}: epochs w/o increase = {2}, AUC = {3}".format(
                epoch+1, epochs, stop_cdn, current_auc) + 20*' ', end='')
            sys.stdout.flush()
        if stop_cdn >= patience:
            if log_to_file:
                print("Patience tolerance reached.", file=log_file)
            elif verbose >= 2:
                print("\nPatience tolerance reached.")
                sys.stdout.flush()
            break
        # Reset model if AUC calculation fails three times
        if stuck_cdn > 2:
            if verbose >= 1:
                print("Training stuck, rolling back to best AUC.",
                      file=log_file)
                sys.stdout.flush()
            model = load_model(model_name)
            stuck_cdn = 0
            epoch = 0
            continue
        epoch += 1
    h5file.close()
    if verbose >= 2:
        print(
            "Training complete. Best validation AUC = {0}".format(best_auc),
            file=log_file)
        sys.stdout.flush()
    if log_file is not sys.stdout:
        log_file.close()
    return model


def test_model(
        model, test_h5_file, model_name='model', batch_size=32, verbose=2,
        show_ROC_curve=True, log_to_file=False):
    """Test model using dataset in train_test_file. Display ROC curve.
    """
    if log_to_file:
        try:
            log_file = open(model_name+'_log.txt', 'a')
        except IOError:
            log_file = open(model_name+'_log.txt', 'w')
    else:
        log_file = sys.stdout
    with h5py.File(test_h5_file, 'r') as h5file:
        if verbose >= 1:
            print("Testing on {0} samples.\nDataset from {1}.".format(
                len(h5file['X_test']), test_h5_file), file=log_file)
            sys.stdout.flush()
        # Score from model loss function
        objective_score = model.evaluate(
            h5file['X_test'], h5file['Y_test'], batch_size=batch_size,
            verbose=0)
        Y_test = h5file['Y_test'][:]
        Y_prob = model.predict_proba(
            h5file['X_test'], batch_size=batch_size, verbose=0)
        Y_prob /= Y_prob.sum(axis=1)[:, np.newaxis]
        Y_pred = model.predict_classes(
            h5file['X_test'], batch_size=batch_size, verbose=0)
    fpr, tpr, thresholds = roc_curve(Y_test[:, 0], Y_prob[:, 0])
    res = 1./len(Y_test)
    inv_curve = np.array(
        [[tp, 1./max(fp, res)]
        for tp,fp in zip(tpr,fpr) if (0.2 <= tp <= 0.8 and fp > 0.)])
    # AUC score
    final_auc = auc(inv_curve[:, 0], inv_curve[:, 1])
    # Number of correct classifications
    accuracy = sum(
        [1 for i in range(len(Y_test)) if Y_test[i, Y_pred[i]] == 1.0])
    # Print results
    if verbose >= 2:
        print("Score    = {0}".format(objective_score), file=log_file)
        print("AUC      = {0}".format(final_auc), file=log_file)
        print("Accuracy = {0}/{1} = {2}\n".format(
            accuracy, len(Y_test), float(accuracy)/len(Y_test)), file=log_file)
        sys.stdout.flush()
    if log_file is not sys.stdout:
        log_file.close()
    if show_ROC_curve:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.plot(inv_curve[:, 0], inv_curve[:, 1])
        ax.set_xlabel("signal efficiency", fontsize=16)
        ax.set_ylabel("1 / backgroud efficiency", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_title("Receiver operating characteristic", fontsize=16)
        fig.tight_layout()
        fig.show()
        return {'score' : objective_score,
                'AUC' : final_auc,
                'accuracy' : float(accuracy)/len(Y_test),
                'ROC curve' : inv_curve}
    else:
        return {'score' : objective_score,
                'AUC' : final_auc,
                'accuracy' : float(accuracy)/len(Y_test)}


def train_test_star_cv(kwargs):
    ikf = kwargs['ikf']
    train_h5_file = kwargs['train_h5_file']
    model_name = kwargs['model_name']
    train_kwargs = kwargs['train_kwargs']
    
    model = load_model(model_name+'_base')
    model_name_ikf = model_name+'_kf{0}'.format(ikf)
    train_model(model, train_h5_file, model_name_ikf, **train_kwargs)
    model = load_model(model_name_ikf)
    return test_model(
        model, train_h5_file, model_name_ikf,
        batch_size=train_kwargs['batch_size'], verbose=train_kwargs['verbose'],
        show_ROC_curve=False, log_to_file=train_kwargs['log_to_file'])


def cross_validate_model(
        model, train_h5_files, model_name='model', batch_size=32, epochs=100,
        patience=10, verbose=2, log_to_file=False, read_into_RAM=False,
        max_jobs=1):
    """Cross validate model using k-folded datasets in train_h5_files.
    Returns lists of scores.
    """
    if isinstance(train_h5_files, str) or len(train_h5_files) < 2:
        print("Number of k-folds must be > 1.")
        return 0
    if max_jobs < 1:
        max_jobs = cpu_count()
    max_jobs = min(max_jobs, cpu_count())
    if max_jobs > 1:
        verbose = min(verbose, 1)
    if log_to_file:
        verbose = 2
    n_folds = len(train_h5_files)
    kf_kwargs = [
        {'ikf' : ikf,
        'train_h5_file' : train_h5_files[ikf],
        'model_name' : model_name,
        'train_kwargs' : {'batch_size' : batch_size,
                          'epochs' : epochs,
                          'patience' : patience,
                          'verbose' : verbose,
                          'log_to_file' : log_to_file,
                          'read_into_RAM' : read_into_RAM}}
        for ikf in xrange(n_folds)]
    save_model(model, model_name+'_base')
    
    if max_jobs > 1:
        pool = Pool(max_jobs)
        results = pool.map(train_test_star_cv, kf_kwargs)
        pool.close()
        pool.join()
        scores = [r['score'] for r in results]
        AUCs = [r['AUC'] for r in results]
        accuracies = [r['accuracy'] for r in results]
    else:
        scores = []
        AUCs = []
        accuracies = []
        for kwargs in kf_kwargs:
            result = train_test_star_cv(kwargs)
            scores.append(result['score'])
            AUCs.append(result['AUC'])
            accuracies.append(result['accuracy'])
    return {'score' : scores,
            'AUC' : AUCs,
            'accuracy' : accuracies}


def compile_model_star_gs(kwargs):
    model_name_igp = kwargs['model_name_igp']
    get_model = kwargs['get_model']
    get_model_args = kwargs['get_model_args']
    optimizer = kwargs['optimizer']
    optimizer_kwargs = kwargs['optimizer_kwargs']
    
    model_igp = get_model(
        *get_model_args, optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs)
    save_model(model_igp, model_name_igp+'_base')


def train_test_star_gs(kwargs):
    igp = kwargs['igp']
    ikf = kwargs['ikf']
    optimizer_kwargs = kwargs['optimizer_kwargs']
    train_h5_file = kwargs['train_h5_file']
    model_name_igp = kwargs['model_name_igp']
    train_kwargs = kwargs['train_kwargs']
    
    if train_kwargs['verbose'] >= 2 and not(train_kwargs['log_to_file']):
        print("Optimizer parameters = {0}, k-fold = {1}".format(
            optimizer_kwargs, ikf))
        sys.stdout.flush()
    model = load_model(model_name_igp+'_base')
    model_name_igp_ikf = model_name_igp+'_kf{0}'.format(ikf)
    train_model(model, train_h5_file, model_name_igp_ikf, **train_kwargs)
    model = load_model(model_name_igp_ikf)
    results = test_model(
        model, train_h5_file, model_name_igp_ikf,
        batch_size=train_kwargs['batch_size'], verbose=train_kwargs['verbose'],
        show_ROC_curve=False, log_to_file=train_kwargs['log_to_file'])
    return {'igp' : igp,
            'ikf' : ikf,
            'parameters' : optimizer_kwargs,
            'results' : results}


def optimizer_grid_search(
        get_model, get_model_args, optimizer, optimizer_kwargs_grid,
        train_h5_files, model_name='model', batch_size=32, epochs=100,
        patience=10, verbose=2, log_to_file=False, read_into_RAM=False,
        max_jobs=1):
    """Perform grid search on optimizer kwargs. Cross validate at each point.
    Returns lists of scores.
    """
    if isinstance(train_h5_files, str) or len(train_h5_files) < 2:
        print("Number of k-folds must be > 1.")
        print("train_h5_files = {0}".format(train_h5_files))
        return 0
    if max_jobs < 1:
        max_jobs = cpu_count()
    max_jobs = min(max_jobs, cpu_count())
    if max_jobs > 1:
        verbose = min(verbose, 1)
    if log_to_file:
        verbose = 2
    # Compile models
    optimizer_kwargs_grid = ParameterGrid(optimizer_kwargs_grid)
    model_kwargs = []
    igp = 0
    for optimizer_kwargs in optimizer_kwargs_grid:
        model_kwargs.append({
            'model_name_igp' : model_name+'_gp{0}'.format(igp),
            'get_model' : get_model,
            'get_model_args' : get_model_args,
            'optimizer' : optimizer,
            'optimizer_kwargs' : optimizer_kwargs})
        igp += 1
    if verbose >= 1:
        print("Compiling models...\n")
        sys.stdout.flush()
    if max_jobs > 1:
        pool = Pool(max_jobs)
        pool.map(compile_model_star_gs, model_kwargs)
        pool.close()
    else:
        for kwargs in model_kwargs:
            compile_model_star_gs(kwargs)
    # Cross-validate models
    train_kwargs = {'batch_size' : batch_size,
                    'epochs' : epochs,
                    'patience' : patience,
                    'verbose' : verbose,
                    'log_to_file' : log_to_file,
                    'read_into_RAM' : read_into_RAM}
    gp_kwargs = []
    igp = 0
    for optimizer_kwargs in optimizer_kwargs_grid:
        model_name_igp = model_name+'_gp{0}'.format(igp)
        ikf = 0
        for train_h5_file in train_h5_files:
            gp_kwargs.append({
                'igp' : igp, 'ikf' : ikf,
                'optimizer_kwargs' : optimizer_kwargs,
                'train_h5_file' : train_h5_file,
                'model_name_igp' : model_name_igp,
                'train_kwargs' : train_kwargs})
            ikf += 1
        igp += 1
    if verbose >= 1:
        print("Cross-validating models...\n")
        sys.stdout.flush()
    if max_jobs > 1:
        pool = Pool(max_jobs)
        results = pool.map(train_test_star_gs, gp_kwargs)
        pool.close()
        pool.join()
    else:
        results = [train_test_star_gs(kwargs) for kwargs in gp_kwargs]
    # Restructure results
    new_results = [
        {'parameters' : {},
         'results' : {'score' : np.zeros(ikf),
                      'AUC' : np.zeros(ikf),
                      'accuracy' : np.zeros(ikf)}}
        for i in xrange(igp)]
    for r in results:
        igp, ikf = r['igp'], r['ikf']
        par = r['parameters']
        res = r['results']
        new_results[igp]['parameters'] = par
        new_results[igp]['results']['score'][ikf] = res['score']
        new_results[igp]['results']['AUC'][ikf] = res['AUC']
        new_results[igp]['results']['accuracy'][ikf] = res['accuracy']
    return new_results


def select_best_model(grid_search_results, metric='AUC', max_is_best=True):
    """Returns parameter values giving best value for metric.
    """
    results_list = [
        (r['parameters'], np.mean(r['results'][metric]))
        for r in grid_search_results]
    results_list.sort(key=lambda x: x[1])
    if max_is_best:
        return {'parameters' : results_list[-1][0],
                metric : results_list[-1][1]}
    else:
        return {'parameters' : results_list[0][0],
                metric : results_list[0][1]}