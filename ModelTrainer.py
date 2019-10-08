import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from sklearn.utils import shuffle
from os import path

# Define the CNN Model
tf.reset_default_graph()
convnet = input_data(shape=[None, 89, 100, 1], name='input')
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1000, activation='relu')
convnet = dropout(convnet, 0.75)

convnet = fully_connected(convnet, 7, activation='softmax')

convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='regression')

model = tflearn.DNN(convnet, tensorboard_verbose=0)
if path.exists("TrainedModel"):
    model.load("TrainedModel/GestureRecogModel.tfl")

loadedImages = []
outputVectors = []

for i in range(0, 1000):
    # Load palm images
    image = cv2.imread('MyDataset/palm/palm_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.resize(gray_image, (320, 120))  # Reduce image size so training can be faster
    loadedImages.append(gray_image.reshape(89, 100, 1))
    outputVectors.append([1, 0, 0, 0, 0, 0, 0])

    # Load fist images
    image = cv2.imread('MyDataset/fist/fist_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.resize(gray_image, (320, 120))  # Reduce image size so training can be faster
    loadedImages.append(gray_image.reshape(89, 100, 1))
    outputVectors.append([0, 1, 0, 0, 0, 0, 0])

for i in range(0, 100):
    i += 1
    # Load ok images
    image = cv2.imread('MyDataset/ok/ok_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.resize(gray_image, (320, 120))  # Reduce image size so training can be faster
    loadedImages.append(gray_image.reshape(89, 100, 1))
    outputVectors.append([0, 0, 1, 0, 0, 0, 0])

    # Load thumb images
    image = cv2.imread('MyDataset/thumb/thumb_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.resize(gray_image, (320, 120))  # Reduce image size so training can be faster
    loadedImages.append(gray_image.reshape(89, 100, 1))
    outputVectors.append([0, 0, 0, 1, 0, 0, 0])

    # Load v images
    image = cv2.imread('MyDataset/v/v_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.resize(gray_image, (320, 120))  # Reduce image size so training can be faster
    loadedImages.append(gray_image.reshape(89, 100, 1))
    outputVectors.append([0, 0, 0, 0, 1, 0, 0])

    # Load l images
    image = cv2.imread('MyDataset/l/l_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.resize(gray_image, (320, 120))  # Reduce image size so training can be faster
    loadedImages.append(gray_image.reshape(89, 100, 1))
    outputVectors.append([0, 0, 0, 0, 0, 1, 0])

    # Load swing images
    image = cv2.imread('MyDataset/swing/swing_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.resize(gray_image, (320, 120))  # Reduce image size so training can be faster
    loadedImages.append(gray_image.reshape(89, 100, 1))
    outputVectors.append([0, 0, 0, 0, 0, 0, 1])


# Shuffle Training Data
loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)

# Train model
model.fit(loadedImages, outputVectors, n_epoch=5, batch_size=64, validation_set=0.1,
          snapshot_step=100, show_metric=True, run_id='convnet_coursera')
model.save("TrainedModel/GestureRecogModel.tfl")
