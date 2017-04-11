import csv
import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

import sklearn.utils
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(ROOT, 'data')


def generator(samples, batch_size=32):
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

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # X_train = np.array(images)
    # print(X_train.shape)
    # y_train = np.array(measurements)

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         nb_val_samples=len(validation_samples), nb_epoch=5)
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
