import sys
from matplotlib import pyplot
import numpy
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, BatchNormalization, Activation, Input, add
import keras
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
	# normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # subtracting pixel mean improves accuracy
    train_norm -= numpy.mean(train_norm, axis=0)
    test_norm -= numpy.mean(train_norm, axis=0)
    # return normalized images
    return train_norm, test_norm

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# define conv layer for RESNET
def define_conv_layer(inputs, num_filters=16, kernel_size=3, strides=1, conv_first=True, batch_normalization=True, activation="relu"):
    conv = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4)
    )
    x = inputs
    if conv_first:
        x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    if not conv_first:
        x = conv(x)
    return x


def define_resnet_model_v1(input_shape, depth):

    # Start model definition.
    num_filters = 16
    num_steps = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    # add initial conv layer
    x = define_conv_layer(
        inputs=inputs, 
        num_filters=num_filters, 
        kernel_size=3, 
        strides=1, 
        conv_first=True,
        batch_normalization=True, 
        activation="relu"
        )

    # define three blocks of convolution layers
    for block in range(3):
        for step in range(num_steps):
            y = define_conv_layer(
                inputs=x,
                num_filters=num_filters,
                kernel_size=3,
                strides = 2 if (block > 0 and step == 0) else 1, # first step downsamples input except for the first block
                conv_first=True,
                batch_normalization=True,
                activation="relu"
            )
            y = define_conv_layer(
                inputs=y,
                num_filters=num_filters,
                kernel_size=3,
                strides = 1,
                conv_first=True,
                batch_normalization=True,
                activation=None
            )
            # after downsample we need to adjust shortcut
            # all other steps use identity shortcut
            if block > 0 and step == 0:
                x = define_conv_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides = 2,
                    conv_first=True,
                    batch_normalization=False,
                    activation=None
                )
            # add shortcut
            x = add([x, y])
            x = Activation("relu")(x)
        # double the number of filters for convolution layers
        num_filters *= 2

    # add classification block
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(
        10, # 10 classes in CIFAR10
        activation='softmax',
        kernel_initializer='he_normal'
    )(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def define_resnet_model_v2(input_shape, depth):

    # Start model definition.
    num_filters = 16
    num_steps = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # add initial conv layer
    x = define_conv_layer(
        inputs=inputs, 
        num_filters=num_filters, 
        kernel_size=3, 
        strides=1, 
        conv_first=True,
        batch_normalization=True, 
        activation="relu"
        )

    # define three blocks of convolution layers
    for block in range(3):
        for step in range(num_steps):
            y = define_conv_layer(
                inputs=x,
                num_filters=num_filters,
                kernel_size=1,
                strides=(2 if (block > 0 and step == 0) else 1), # first step downsamples input except for the first block
                conv_first=False,
                batch_normalization=(False if (block == 0 and step == 0) else True),
                activation=(None if (block == 0 and step == 0) else "relu")
            )
            y = define_conv_layer(
                inputs=y,
                num_filters=num_filters,
                kernel_size=3,
                strides = 1,
                conv_first=False,
                batch_normalization=True,
                activation="relu"
            )
            y = define_conv_layer(
                inputs=y,
                num_filters=(num_filters * 4 if block == 0 else num_filters * 2),
                kernel_size=1,
                strides=1,
                conv_first=False,
                batch_normalization=True,
                activation="relu"
            )
            # after downsample we need to adjust shortcut
            # all other steps use identity shortcut
            if step == 0:
                x = define_conv_layer(
                    inputs=x,
                    num_filters=(num_filters * 4 if block == 0 else num_filters * 2),
                    kernel_size=1,
                    strides = (2 if (block > 0 and step == 0) else 1),
                    conv_first=False,
                    batch_normalization=False,
                    activation=None
                )
            # add shortcut
            x = add([x, y])

        # increase the number of filters for convolution layers
        num_filters = (num_filters * 4 if block == 0 else num_filters * 2)

    # add classification block
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(
        10, # 10 classes in CIFAR10
        activation='softmax',
        kernel_initializer='he_normal'
    )(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# run the test harness for evaluating a model
def run_test_harness(model_depth=20, num_epochs=10, version="v1"):
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # create RESNET model
    if version == "v1":
        model = define_resnet_model_v1(trainX.shape[1:], model_depth)
    else:
        model = define_resnet_model_v2(trainX.shape[1:], model_depth)
    # compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy']
    )
    model.summary()
	# fit model
    history = model.fit(trainX, trainY, epochs=num_epochs, batch_size=64, validation_data=(testX, testY))
	# evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
	# learning curves
    summarize_diagnostics(history)

# run RESNET20
run_test_harness(
    model_depth=20, 
    num_epochs=20,
    version="v1"
)

# run RESNET56
#run_test_harness(
#    model_depth=56, 
#    num_epochs=20,
#    version="v2"
#)








