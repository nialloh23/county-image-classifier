#!/usr/bin/python

"""Standard CNN Network."""

from typing import Tuple

from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, Model


def cnn_network(kernel_size: int = 3) -> Model:
    """Return CNN Keras Model."""
    
    model = Sequential()
        
    model.add(Conv2D(32, (kernel_size, kernel_size), activation='relu', input_shape=(300, 300, 3)))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(64, (kernel_size, kernel_size), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(128, (kernel_size, kernel_size), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(128, (kernel_size, kernel_size), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    
    print(model.summary())
    return model

if __name__ == '__main__':
    cnn_network()