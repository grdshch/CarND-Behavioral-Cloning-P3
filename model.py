import csv
import cv2
import numpy as np
import os
import sys



ROOT = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(ROOT, 'train')

if __name__ == '__main__':
    with open(os.path.join(DATA, 'driving_log.csv')) as csvfile:
        lines = [line for line in csv.reader(csvfile)][1:]
    images = []
    measurements = []
    for line in lines:
        source = os.path.join(DATA, line[0])
        image = cv2.imread(source)
        images.append(image)
        measurements.append(line[3])

    X_train = np.array(images)
    print(X_train.shape)
    y_train = np.array(measurements)

    from keras.models import Sequential
    from keras.layers import Flatten, Dense

    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=6)
    model.save('model.h5')
