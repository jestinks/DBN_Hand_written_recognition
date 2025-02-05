import numpy as np
import os
import cv2
import pickle
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, y_train, x_test, y_test

def save_dbn_model(dbn, filename='dbn_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(dbn, f)

def load_dbn_model(filename='dbn_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def preprocess_custom_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 784)
    return img

def process_custom_input(image, model):
    processed_img = preprocess_custom_image(image)
    hidden, reconstruction = model.get_hidden_and_reconstruction(processed_img)
    return processed_img.reshape(28, 28), hidden.reshape(10, 10), reconstruction.reshape(28, 28)
