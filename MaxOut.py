#!/usr/bin/python

from JetPreprocessing import ShowJetImage
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, MaxoutDense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn import cross_validation
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

optimizer = Adam(lr=0.01)
stopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

X      = []
Y      = []
#W_im   = np.zeros((25,25))
#QCD_im = np.zeros((25,25))
for fname in os.listdir('WJetImages'):
    with open('WJetImages/{0}'.format(fname), 'r') as f:
        im    = np.load(f)
        #W_im += im
        X.append(im.flatten())
        Y.append([1.0, 0.0])
for fname in os.listdir('QCDJetImages'):
    with open('QCDJetImages/{0}'.format(fname), 'r') as f:
        im      = np.load(f)
        #QCD_im += im
        X.append(im.flatten())
        Y.append([0.0, 1.0])

#ShowJetImage(2.0*W_im / len(X))
#ShowJetImage(2.0*QCD_im / len(X))

X = np.array(X).astype('float32')
Y = np.array(Y).astype('float32')
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

model.fit(X_train, Y_train, nb_epoch=epochs, batch_size=batch_size,
          validation_split=1.0/8.0, callbacks=[stopper], verbose=2)

objective_score = model.evaluate(X_test, Y_test, batch_size=batch_size)

print objective_score