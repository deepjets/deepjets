import h5py
import matplotlib.pyplot as plt
import numpy as np
from .models import save_model
from .utils import load_images
from keras.optimizers import Adam
from sklearn import cross_validation
from sklearn.metrics import auc, roc_curve


def prepare_datasets(sig_h5_file, bkd_h5_file, aux_vars=[], out_file_name='train_test',
                     n_sig=-1, n_bkd=-1, n_folds=2, val_frac=0.1,
                     shuffle=False, shuffle_seed=1):
    """Combine signal, background images; split into k-folded training, validation, test sets.
    
    Returns list of filenames for each k-fold.
    TODO: test support for additional fields.
    """
    # Load images
    sig_images, sig_aux_data = load_images(sig_h5_file, n_sig, aux_vars, shuffle)
    bkd_images, bkd_aux_data = load_images(bkd_h5_file, n_bkd, aux_vars, shuffle)
    n_sig = len(sig_images)
    n_bkd = len(bkd_images)
    n_images = n_sig + n_bkd
    images = np.concatenate((sig_images, bkd_images))
    images = images.reshape(-1, images.shape[1] * images.shape[2])
    aux_data = {var : np.concatenate((sig_aux_data[var], bkd_aux_data[var])) for var in aux_vars}
    # True classes
    classes = np.concatenate([np.repeat([[1, 0]], n_sig, axis=0),
                              np.repeat([[0, 1]], n_bkd, axis=0)])
    kf = cross_validation.KFold(n_images, n_folds,
                                shuffle=True, random_state=shuffle_seed)
    i = 0
    files = []
    for train, test in kf:
        out_file = out_file_name + '_{0}.h5'.format(i)
        # Shuffle to make sure validation set contains both classes
        np.random.shuffle(train)
        with h5py.File(out_file, 'w') as h5file:
            n_val = int(val_frac * len(train))
            h5file.create_dataset('X_train', data=images[train][:n_val])
            h5file.create_dataset('Y_train', data=classes[train][:n_val])
            for var in aux_vars:
                h5file.create_dataset(var + '_train', data=aux_data[var][train][:n_val])
            h5file.create_dataset('X_val', data=images[train][n_val:])
            h5file.create_dataset('Y_val', data=classes[train][n_val:])
            for var in aux_vars:
                h5file.create_dataset(var + '_val', data=aux_data[var][train][n_val:])
            h5file.create_dataset('X_test', data=images[test])
            h5file.create_dataset('Y_test', data=classes[test])
            for var in aux_vars:
                h5file.create_dataset(var + '_test', data=aux_data[var][test])
        files.append(out_file)
        i += 1
    return files


def train_model(model, train_test_file, batch_size,
                epochs=100, patience=10, loss='categorical_crossentropy', optimizer=Adam(),
                name='unnamed'):
    """Train model using train, val datasets in train_test_file. Save best AUC using name.
    """
    model.compile(loss=loss, optimizer=optimizer)
    best_auc = 0.
    stop_cdn = 0
    for epoch in range(epochs):
        print "Epoch {0}/{1}...".format(epoch + 1, epochs)
        with h5py.File(train_test_file, 'r') as h5file:
            # Fitting
            model.fit(h5file['X_train'], h5file['Y_train'], batch_size=batch_size, verbose=0,
                      nb_epoch=1, shuffle='batch')
            # Validation
            Y_val = h5file['Y_val'][:]
            Y_prob = model.predict_proba(h5file['X_val'], batch_size=batch_size, verbose=0)
            Y_prob /= Y_prob.sum(axis=1)[:, np.newaxis]
        # Calculate AUC for custom early stopping
        fpr, tpr, _ = roc_curve(Y_val[:, 0], Y_prob[:, 0])
        res = 1. / len(Y_val)
        inv_curve = np.array(
            [[tp, 1. / max(fp, res)]
            for tp,fp in zip(tpr,fpr) if (0.2 <= tp <= 0.8 and fp > 0.)])
        current_auc = auc(inv_curve[:, 0], inv_curve[:, 1])
        if current_auc > best_auc:
            best_auc = current_auc
            stop_cdn = 0
            save_model(model, name)
        else:
            stop_cdn += 1
        print "Epochs w/o increase = {0}, AUC = {1}".format(stop_cdn, current_auc)
        if stop_cdn >= patience:
            print "Patience tolerance reached"
            break
    print "Training complete"


def test_model(model, train_test_file, batch_size):
    """Test model using test dataset in train_test_file. Show ROC curve.
    """
    with h5py.File(train_test_file, 'r') as h5file:
        # Score from model loss function
        objective_score = model.evaluate(h5file['X_test'], h5file['Y_test'],
                                         batch_size=batch_size, verbose=0)
        Y_test = h5file['Y_test'][:]
        Y_prob = model.predict_proba(h5file['X_test'], batch_size=batch_size, verbose=0)
        Y_prob /= Y_prob.sum(axis=1)[:, np.newaxis]
        Y_pred = model.predict_classes(h5file['X_test'], batch_size=batch_size, verbose=0)
    fpr, tpr, thresholds = roc_curve(Y_test[:, 0], Y_prob[:, 0])
    res = 1. / len(Y_test)
    inv_curve = np.array(
        [[tp, 1. / max(fp, res)]
        for tp,fp in zip(tpr,fpr) if (0.2 <= tp <= 0.8 and fp > 0.)])
    # AUC score
    final_auc = auc(inv_curve[:, 0], inv_curve[:, 1])
    # Number of correct classifications
    accuracy = sum([1 for i in range(len(Y_test)) if Y_test[i, Y_pred[i]] == 1.0])

    print "Score    = {0}".format(objective_score)
    print "AUC      = {0}".format(final_auc)
    print "Accuracy = {0}/{1} = {2}".format(
        accuracy, len(Y_test), float(accuracy) / len(Y_test) )
    plt.figure()
    plt.plot(inv_curve[:, 0], inv_curve[:, 1])
    plt.xlabel("signal efficiency")
    plt.ylabel("(backgroud efficiency)$^{-1}$")
    plt.title("Receiver operating characteristic")
    plt.show()


def train_old(model,
              signal_files, background_files,
              epochs=100, patience=10, batch_size=32, flatten=False):
    """
    TODO: update with James' new code
    """
    X = []
    y = []
    for fname in signal_files:
        with h5py.File(fname, 'r') as infile:
            images = infile['images'][:10000]
            if flatten:
                images = images.reshape(-1, images.shape[1] * images.shape[2])
            X.append(images)
            y.append(np.repeat([[1, 0]], images.shape[0], axis=0))
    for fname in background_files:
        with h5py.File(fname, 'r') as infile:
            images = infile['images'][:10000]
            if flatten:
                images = images.reshape(-1, images.shape[1] * images.shape[2])
            X.append(images)
            y.append(np.repeat([[0, 1]], images.shape[0], axis=0))
    X = np.concatenate(X)
    y = np.concatenate(y)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    stopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
    hist = model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size,
                     validation_split=1./8., callbacks=[stopper], verbose=2)
    # TODO: plot hist.history
    print model.evaluate(X_test, y_test, batch_size=batch_size)

