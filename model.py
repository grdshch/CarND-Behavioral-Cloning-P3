import csv
import cv2
import numpy as np
import os
import sys

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.utils import plot_model

import sklearn.utils
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(ROOT, 'data_both')

BATCH_SIZE = 32


def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source = os.path.join(DATA, batch_sample[0])
                center_image = cv2.imread(source)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


if __name__ == '__main__':
    with open(os.path.join(DATA, 'driving_log.csv')) as csvfile:
        samples = [line for line in csv.reader(csvfile)][1:]

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    model = Sequential()
    model.add(Cropping2D(cropping=((70, 0), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, ))
    model.add(Conv2D(filters=6, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=16, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')


    plot_model(model, to_file='model.png', show_shapes=True)

    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=len(train_samples) / BATCH_SIZE,
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_samples) / BATCH_SIZE,
                                         epochs=10)
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save('model.h5')
