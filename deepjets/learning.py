from __future__ import print_function
import h5py
import numpy as np
import sys
from .models import load_model, save_model
from .utils import likelihood_ratio, plot_roc_curve
from keras import callbacks
from multiprocessing import Pool, cpu_count
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import auc, roc_curve


def train_model(
        model, train_h5_file, model_name='model', batch_size=100, epochs=100,
        val_frac=0.1, patience=10, lr_init=0.001, lr_scale_factor=1.,
        custom_lr_schedule=None, verbose=1, log_to_file=False,
        read_into_ram=False):
    """Train model. Save model with best score.
    
    Args:
        model: keras model to train.
        train_h5_file: name of h5 file containing 'X_train', 'Y_train',
                       'weights_train' (optional) datasets.
        model_name: base filename to use for saving models.
        batch_size, epochs: integers to pass to keras.
        val_frac: fraction of training data held out for validation.
        patience: number of epochs to run without improvement.
        lr_init: initial learning rate.
        lr_scale_factor: learning rate multiplied by this each epoch.
        custom_lr_schedule: function that takes epoch number (int) and returns
                            learning rate (float). Overrides lr_init and
                            lr_scale_factor.
        verbose: 0 suppresses all progress updates.
                 1 provides fancy progress updates.
                 2 provides simple progress updates.
        log_to_file: if True full progress updates are written to
                     model_name_log.txt.
        read_into_ram: if True images read into RAM, otherwise h5 dataset
                       passed directly to keras.
    Returns:
        history: keras object containing model and training history.
    """
    save_model(model, model_name)
    if log_to_file:
        log_file = open(model_name+'_log.txt', 'w')
    else:
        log_file = sys.stdout
    print("Using loss function for early stopping.", file=log_file)
    h5file = h5py.File(train_h5_file, 'r')
    if verbose >= 1 or log_to_file:
        print("Datasets from {0}.".format(train_h5_file), file=log_file)
        print("epochs = {0}, ".format(epochs), file=log_file)
        print("val_frac = {0}, ".format(val_frac), file=log_file)
        print("patience = {0}, ".format(patience), file=log_file)
        if custom_lr_schedule is None:
            print("lr_init = {0}, ".format(lr_init), file=log_file)
            print("lr_scale_factor = {0}, ".format(lr_scale_factor),
                  file=log_file)
        else:
            print("Using custom learning rate schedule function.",
                  file=log_file)
        sys.stdout.flush()
    if log_file is not sys.stdout:
        log_file.close()
    if read_into_ram:
        shuffle = False
        X_train = h5file['X_train'][:]
        Y_train = h5file['Y_train'][:]
        try:
            weights_train = h5file['auxvars_train']['weights'][:]
        except KeyError:
            weights_train = None
    else:
        shuffle = 'batch'
        X_train = h5file['X_train']
        Y_train = h5file['Y_train']
        try:
            weights_train = h5file['auxvars_train']['weights']
        except KeyError:
            weights_train = None
    checkpointer = callbacks.ModelCheckpoint(
        '{0}_weights.h5'.format(model_name), monitor='val_loss',
        verbose=verbose, save_best_only=True, mode='auto')
    earlystopper = callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, verbose=verbose, mode='auto')
    if custom_lr_schedule is None:
        lr_s = lambda i: float(lr_init*(lr_scale_factor**i))
    scheduler = callbacks.LearningRateScheduler(lr_s)
    callback_list = [checkpointer, earlystopper, scheduler]
    history = model.fit(
        X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
        validation_split=val_frac, verbose=verbose,
        sample_weight=weights_train, shuffle=shuffle, callbacks=callback_list)
    h5file.close()
    if log_to_file:
        log_file = open(model_name+'_log.txt', 'a')
        print("\nTraining history:\n", file=log_file)
        print(history.history, file=log_file)
        log_file.close()
    return history
    
    
def train_model_auc_score(
        model, train_h5_file, model_name='model', batch_size=100, epochs=100,
        patience=10, lr_init=0.0001, lr_scale_factor=1.,
        custom_lr_schedule=None, verbose=2, log_to_file=False,
        read_into_ram=False):
    """Train model. Save model with best ROC AUC score.
    
    Args:
        model: keras model to train.
        train_h5_file: name of h5 file containing 'X_train', 'Y_train',
                       'weights_train' (optional) datasets.
        model_name: base filename to use for saving models.
        batch_size, epochs: integers to pass to keras.
        patience: number of epochs to run without improvement.
        lr_init: initial learning rate.
        lr_scale_factor: learning rate multiplied by this each epoch.
        custom_lr_schedule: function that takes epoch number (int) and returns
                            learning rate (float). Overrides lr_init and
                            lr_scale_factor.
        verbose: 0 suppresses all progress updates.
                 1 provides simple progress updates.
                 2 provides detailed progress updates.
        log_to_file: if True full progress updates are written to
                     model_name_log.txt.
        read_into_ram: if True images read into RAM, otherwise h5 dataset
                       passed directly to keras.
    Returns:
        model: model at final point in training (usually not the best model).
    """
    save_model(model, model_name)
    if log_to_file:
        log_file = open(model_name+'_log.txt', 'w')
    else:
        log_file = sys.stdout
    epoch = 0
    stop_cdn = 0
    best_score = 0
    print("Using ROC AUC for early stopping.", file=log_file)
    h5file = h5py.File(train_h5_file, 'r')
    if verbose >= 1 or log_to_file:
        print("Training on {0} samples, ".format(len(h5file['X_train'])) + 
              "validating on {0} samples.".format(len(h5file['X_val'])),
              file=log_file)
        print("Datasets from {0}.".format(train_h5_file), file=log_file)
        print("epochs = {0}, ".format(epochs), file=log_file)
        print("patience = {0}, ".format(patience), file=log_file)
        if custom_lr_schedule is None:
            print("lr_init = {0}, ".format(lr_init), file=log_file)
            print("lr_scale_factor = {0}, ".format(lr_scale_factor),
                  file=log_file)
        else:
            print("Using custom learning rate schedule function.",
                  file=log_file)
        sys.stdout.flush()
    if log_file is not sys.stdout:
        log_file.close()
    if read_into_ram:
        shuffle = False
        X_train = h5file['X_train'][:]
        X_val = h5file['X_val'][:]
    else:
        shuffle = 'batch'
        X_train = h5file['X_train']
        X_val = h5file['X_val']
    try:
        weights_train = h5file['auxvars_train']['weights'][:]
    except KeyError:
        weights_train = None
    try:
        weights_val = h5file['auxvars_val']['weights'][:]
    except KeyError:
        weights_val = None
    Y_train = h5file['Y_train'][:]
    Y_val = h5file['Y_val'][:]
    if custom_lr_schedule is None:
        lr_s = lambda i: float(lr_init*(lr_scale_factor**i))
    while epoch < epochs:
        # Fitting and validation
        model.optimizer.lr.set_value(lr_s(epoch))
        model.fit(
            X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=0,
            sample_weight=weights_train, shuffle=shuffle)
        # Calculate AUC for custom early stopping
        Y_prob = model.predict_proba(
            X_val, batch_size=batch_size, verbose=0)
        lklhd_rat, bins = likelihood_ratio(
            Y_val, Y_prob[:, 0], weights_val)
        scores = lklhd_rat[np.digitize(Y_prob[:, 0], bins) - 1]
        fpr, tpr, _ = roc_curve(
            Y_val[:, 0], scores, sample_weight=weights_val)
        current_score = auc(fpr, tpr)
        if current_score > best_score:
            best_score = current_score
            stop_cdn = 0
            save_model(model, model_name)
        else:
            stop_cdn += 1
        if log_to_file:
            log_file = open(model_name+'_log.txt', 'a')
            print(
                "Epoch {0}/{1}: ".format(epoch+1, epochs) +
                "epochs w/o improvement = {0}, ".format(stop_cdn) +
                "score = {0}".format(current_score), file=log_file)
            log_file.close()
        elif verbose >= 2:
            print("\r", end='')
            print(
                "Epoch {0}/{1}: ".format(epoch+1, epochs) +
                "epochs w/o improvement = {0}, ".format(stop_cdn) +
                "score = {0}".format(current_score) + 20*" ", end='')
            sys.stdout.flush()
        if stop_cdn >= patience:
            if log_to_file:
                log_file = open(model_name+'_log.txt', 'a')
                print("Patience tolerance reached.", file=log_file)
                log_file.close()
            elif verbose >= 2:
                print("\nPatience tolerance reached.")
                sys.stdout.flush()
            break
        epoch += 1
    h5file.close()
    if log_to_file:
        log_file = open(model_name+'_log.txt', 'a')
        print(
            "Training complete. Best validation score = {0}".format(
                best_score),
            file=log_file)
        log_file.close()
    elif verbose >= 2:
        print(
            "\nTraining complete. Best validation score = {0}".format(
                best_score),
            file=log_file)
        sys.stdout.flush()
    return model


def test_model(
        model, test_h5_file, model_name='model', batch_size=100, verbose=2,
        log_to_file=False, show_roc_curve=True, X_dataset='X_test',
        Y_dataset='Y_test'):
    """Test model. Display ROC curve.
    
    Args:
        model: keras model to test.
        test_h5_file: name of h5 file containing test datasets.
        model_name: base filename to use for saving models.
        batch_size: integer to pass to keras.
        verbose: 0 suppresses all progress updates.
                 1 provides minimal progress updates.
                 2 provides full progress updates.
        log_to_file: if True full progress updates are written to
                     model_name_log.txt.
        show_roc_curve: if True plot, display and output ROC curve.
        X_dataset: name of X_test dataset.
        Y_dataset: name of Y_test dataset.
    Returns:
        dict of scores and optionally ROC curve data.
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
                len(h5file[X_dataset]), test_h5_file), file=log_file)
            sys.stdout.flush()
        try:
            weights_test = h5file['auxvars_test']['weights'][:]
        except KeyError:
            weights_test = None
        Y_test = h5file[Y_dataset][:]
        # Score from model loss function
        objective_score = model.evaluate(
            h5file[X_dataset], Y_test, batch_size=batch_size, verbose=0,
            sample_weight=weights_test)
        Y_prob = model.predict_proba(
            h5file[X_dataset], batch_size=batch_size, verbose=0)
        Y_pred = model.predict_classes(
            h5file[X_dataset], batch_size=batch_size, verbose=0)
    # Calculate inverse ROC curve and AUC
    lklhd_rat, bins = likelihood_ratio(Y_test, Y_prob[:, 0], weights_test)
    scores = lklhd_rat[np.digitize(Y_prob[:, 0], bins) - 1]
    fpr, tpr, _ = roc_curve(Y_test[:, 0], scores, sample_weight=weights_test)
    auc_score = auc(fpr, tpr)
    res = 1./len(Y_test)
    inv_roc_curve =  np.array([[tp, 1./max(fp, res)]
                               for tp,fp in zip(tpr,fpr)
                               if (0.2 <= tp <= 0.8 and fp > 0.)])
    # Number of correct classifications
    accuracy_score = sum(
        [1 for i in range(len(Y_test)) if Y_test[i, Y_pred[i]] == 1.0])
    # Print results
    if verbose >= 2:
        print("Score    = {0}".format(objective_score), file=log_file)
        print("ROC AUC  = {0}".format(auc_score), file=log_file)
        print(
            "Accuracy = {0}/{1} = {2}\n".format(
                accuracy_score, len(Y_test),
                float(accuracy_score)/len(Y_test)),
            file=log_file)
        sys.stdout.flush()
    if log_file is not sys.stdout:
        log_file.close()
    if show_roc_curve:
        plot_roc_curve(inv_roc_curve)
    return {'score' : objective_score,
            'AUC' : auc_score,
            'accuracy' : float(accuracy_score)/len(Y_test),
            'ROC_curve' : inv_roc_curve}


def train_test_star_cv(kwargs):
    """Helper function for cross validating in parallel.
    
    Train and test model on k-fold. Return test results."""
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
        show_roc_curve=False, log_to_file=train_kwargs['log_to_file'])


def cross_validate_model(
        model, train_h5_files, model_name='model', batch_size=100, epochs=100,
        val_frac=0.1, patience=10, lr_init=0.001, lr_scale_factor=1.,
        custom_lr_schedule=None, verbose=1, log_to_file=False,
        read_into_ram=False, max_jobs=1):
    """Cross validate model using k-folded datasets.
    
    Args:
        model: keras model to train.
        train_h5_files: list of h5 files containing k-folds to train on. Each
                        file contains 'X_train', 'Y_train', 'weights_train'
                        (optional) datasets.
        model_name: base filename to use for saving models.
        batch_size, epochs: integers to pass to keras.
        val_frac: fraction of training data held out for validation.
        patience: number of epochs to run without improvement.
        lr_init: initial learning rate.
        lr_scale_factor: learning rate multiplied by this each epoch.
        custom_lr_schedule: function that takes epoch number (int) and returns
                            learning rate (float). Overrides lr_init and
                            lr_scale_factor.
        verbose: 0 suppresses all progress updates.
                 1 provides fancy progress updates.
                 2 provides simple progress updates.
        log_to_file: if True full progress updates are written to
                     model_name_log.txt.
        read_into_ram: if True images read into RAM, otherwise h5 dataset
                       passed directly to keras.
        max_jobs: number of processors to utilise.
    Returns:
        dict of test results listed by k-fold. 
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
                           'val_frac' : val_frac,
                           'patience' : patience,
                           'lr_init' : lr_init,
                           'lr_scale_factor' : lr_scale_factor,
                           'custom_lr_schedule' : custom_lr_schedule,
                           'verbose' : verbose,
                           'log_to_file' : log_to_file,
                           'read_into_ram' : read_into_ram}}
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
    """Helper function for grid searching in parallel.
    
    Compiles and saves models for all values of optimizer parameters."""
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
    """Helper function for grid searching in parallel.
    
    Train and test model on grid point, k-fold. Return test results,
    grid point and k-fold indices."""
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
        show_roc_curve=False, log_to_file=train_kwargs['log_to_file'])
    return {'igp' : igp,
            'ikf' : ikf,
            'parameters' : optimizer_kwargs,
            'results' : results}


def optimizer_grid_search(
        get_model, get_model_args, optimizer, optimizer_kwargs_grid,
        train_h5_files, model_name='model', batch_size=100, epochs=100,
        val_frac=0.1, patience=10, lr_init=0.001, lr_scale_factor=1.,
        custom_lr_schedule=None, verbose=1, log_to_file=False,
        read_into_ram=False, max_jobs=1):
    """Perform cross-validated grid search on optimizer kwargs.
    
    Args:
        get_model: function to generate keras models.
        get_model_args: list of arguments to pass to get_model.
        optimizer: keras optimizer to use in training model.
        optimizer_kwargs_grid: dict defining ranges for optimizer parametes to
                               be varied.
        train_h5_files: list of h5 files containing k-folds to train on. Each
                        file contains 'X_train', 'Y_train', 'X_val', 'Y_val',
                        'weights_train' (optional) datasets.
        model_name: base filename to use for saving models.
        batch_size, epochs: integers to pass to keras.
        val_frac: fraction of training data held out for validation.
        patience: number of epochs to run without improvement.
        lr_init: initial learning rate.
        lr_scale_factor: learning rate multiplied by this each epoch.
        custom_lr_schedule: function that takes epoch number (int) and returns
                            learning rate (float). Overrides lr_init and
                            lr_scale_factor.
        verbose: 0 suppresses all progress updates.
                 1 provides fancy progress updates.
                 2 provides simple progress updates.
        log_to_file: if True full progress updates are written to
                     model_name_log.txt.
        read_into_ram: if True images read into RAM, otherwise h5 dataset
                       passed directly to keras.
        max_jobs: number of processors to utilise.
    Returns:
        list of dicts of goptimizer parameters, test results listed by k-fold. 
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
                    'val_frac' : val_frac,
                    'patience' : patience,
                    'lr_init' : lr_init,
                    'lr_scale_factor' : lr_scale_factor,
                    'custom_lr_schedule' : custom_lr_schedule,
                    'verbose' : verbose,
                    'log_to_file' : log_to_file,
                    'read_into_ram' : read_into_ram}
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


def select_best_model(grid_search_results, metric='score', max_is_best=False):
    """Select parameter values giving best value for metric.
    
    Args:
        grid_search_results: results from optimizer_grid_search function.
        metric: name of metric to order results by.
        max_is_best: if True try and maximize metric, else minimize.
    Return:
        dict containing parameter and metric values for best grid point.
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