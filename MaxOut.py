#!/usr/bin/python

from JetPreprocessing import ShowJetImage
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, MaxoutDense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn import cross_validation
import h5py
import numpy as np
import os

epochs     = 100
patience   = 10
batch_size = 32

model = Sequential()

model.add(MaxoutDense(input_shape=(625,), output_dim=256, nb_feature=5, init='he_uniform'))
model.add(MaxoutDense(output_dim=128, nb_feature=5))
model.add(Dense(output_dim=64, activation='relu'))
model.add(Dense(output_dim=25, activation='relu'))
model.add(Dense(output_dim=2, activation='sigmoid'))

optimizer = Adam()
stopper   = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

max_n = 10000

X = []
Y = []

with h5py.File('wprime_images.h5', 'r') as infile:
    n_s    = min(len(infile['images']['image']), max_n)
    images = infile['images']['image'][:n_s]
    images = images.reshape(-1, images.shape[1] * images.shape[2])
    X.append(images)
    Y.append(np.repeat([[1, 0]], images.shape[0], axis=0))
with h5py.File('qcd_images.h5', 'r') as infile:
    n_s    = min(len(infile['images']['image']), max_n)
    images = infile['images']['image'][:n_s]
    images = images.reshape(-1, images.shape[1] * images.shape[2])
    X.append(images)
    Y.append(np.repeat([[0, 1]], images.shape[0], axis=0))

X = np.concatenate(X)
Y = np.concatenate(Y)

"""
for fname in os.listdir('WJetImages'):
    with open('WJetImages/{0}'.format(fname), 'r') as f:
        im = np.load(f)
        X.append(im.flatten())
        Y.append([1.0, 0.0])
for fname in os.listdir('QCDJetImages'):
    with open('QCDJetImages/{0}'.format(fname), 'r') as f:
        im = np.load(f)
        X.append(im.flatten())
        Y.append([0.0, 1.0])
"""

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.1)

model.fit(X_train, Y_train, nb_epoch=epochs, batch_size=batch_size,
          validation_split=0.1, callbacks=[stopper], verbose=2)

objective_score = model.evaluate(X_test, Y_test, batch_size=batch_size)
classes         = model.predict_classes(X_test, batch_size=batch_size)
accuracy        = [1 for i in range(len(Y_test)) if Y_test[i,classes[i]] == 1.0]

print "\nScore    = {0}".format(objective_score)
print   "Accuracy = {0}/{1}".format(sum(accuracy), len(Y_test))