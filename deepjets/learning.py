from sklearn.grid_search import ParameterGrid
from sklearn import cross_validation

from keras.callbacks import EarlyStopping

import numpy as np
import h5py


def train(model,
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

