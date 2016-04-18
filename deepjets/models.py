from keras.layers.core import Dense, MaxoutDense, Dropout, Activation
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
# https://groups.google.com/forum/#!topic/keras-users/8Ncd0dpuPNE
# BatchNormalization is preferred over LRN
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.regularizers import l2


def draw_model(model, model_name):
    # Plot the network (requires pydot and graphviz)
    from keras.utils.visualize_util import to_graph
    open('model.svg', 'w').write(
        to_graph(model).create(prog='dot', format='svg'))


def save_model(model, model_name, overwrite=True):
    """Save model architecture and weights."""
    json_string = model.to_json()
    open('{0}_arch.json'.format(model_name), 'w').write(json_string)
    model.save_weights('{0}_weights.h5'.format(model_name), overwrite)


def load_model(model_name, compile=True, **kwargs):
    """Load model architecture and weights."""
    model = model_from_json(open('{0}_arch.json'.format(model_name)).read())
    model.load_weights('{0}_weights.h5'.format(model_name))
    if compile:
        compile_model(model, **kwargs)
    return model


def compile_model(model, loss='categorical_crossentropy', optimizer=Adam, optimizer_kwargs={}):
    optimizer = optimizer(**optimizer_kwargs)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def get_maxout(
        size, loss='categorical_crossentropy', optimizer=Adam,
        optimizer_kwargs={}):
    # MaxOut network
    model = Sequential()
    model.add(MaxoutDense(256, input_shape=(size,), nb_feature=5,
                          init='he_uniform'))
    model.add(MaxoutDense(128, nb_feature=5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(25))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    optimizer = optimizer(**optimizer_kwargs)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def get_convnet():
    # ConvNet architecture
    # [Dropout -> Conv -> ReLU -> MaxPool] * 3 -> LRN ->
    # [Dropout -> FC -> ReLU] -> Dropout -> Sigmoid
    model = Sequential()
    model.add(ZeroPadding2D())
    # [Dropout -> Conv -> ReLU -> MaxPool] * 3
    for filtersize, poolsize in [(11, 2), (3, 3), (3, 3)]:
        model.add(Dropout(0.2))
        model.add(Convolution2D(32, 1, filtersize, filtersize,
                                # init='he_uniform' ?
                                W_regularizer=l2()))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(poolsize=(poolsize, poolsize)))
    # -> LRN
    model.add(BatchNormalization())
    # -> [Dropout -> FC -> ReLU]
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    # -> Dropout -> Sigmoid
    model.add(Dropout(0.1))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    return model
