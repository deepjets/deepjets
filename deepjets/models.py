from keras.models import Sequential
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation
# https://groups.google.com/forum/#!topic/keras-users/8Ncd0dpuPNE
# BatchNormalization is preferred over LRN
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2


def draw_model(model, name):
    # Plot the network (requires pydot and graphviz)
    from keras.utils.visualize_util import to_graph
    open('model.svg', 'w').write(to_graph(model).create(prog='dot', format='svg'))


def save_model(model, name):
    json_string = model.to_json()
    open('model_{0}_arch.json'.format(name), 'w').write(json_string)
    model.save_weights('model_{0}_weights.h5'.format(name))


def load_model(name):
    model = model_from_json(open('model_{0}_arch.json'.format(name)).read())
    model.load_weights('model_{0}_weights.h5'.format(name))
    return model


def get_maxout(size):
    # MaxOut network
    model = Sequential()
    model.add(MaxoutDense(256, input_shape=(size**2,), nb_feature=5,
                          init='he_uniform'))
    model.add(MaxoutDense(128, nb_feature=5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(25))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    return model


def get_convnet():
    # ConvNet architecture
    # [Dropout -> Conv -> ReLU -> MaxPool] * 3 -> LRN -> [Dropout -> FC -> ReLU] -> Dropout -> Sigmoid
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
