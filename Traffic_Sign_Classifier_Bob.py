# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 08:55:26 2018

@author: Administrator
"""

### Step 0:Load the data
# Load pickled data
import pickle
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import os, sys
import matplotlib.image as mpimg
from PIL import Image
from itertools import compress
import random
import math
import cv2

# TODO: Fill this in based on where you saved the training and testing data

training_file = ".\\traffic-signs-data\\train.p"
validation_file= ".\\traffic-signs-data\\valid.p"
testing_file = ".\\traffic-signs-data\\test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas
# TODO: Number of training examples
assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
#pad images with 0s
X_train = np.pad(X_train,((0,0),(2,2),(2,2),(0,0)),'constant')
X_valid = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = y_train[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
#----------------------------------------------------------------------------------

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

# %matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

X_train, y_train = shuffle(X_train, y_train)

def grayscale(image):
    logits = tf.image.rgb_to_grayscale(image)
    return logits

def normalize(image, min_val, max_val, min_color=0, max_color=255):
    """ Normalize image colors between min and max values
    
    a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
    """
    output = tf.divide(
        tf.multiply(tf.subtract(image, min_color), tf.subtract(max_val, min_val)),
        (max_color - min_color)
    )
    return output

def preprocess(inputs):
    logits = grayscale(inputs)
    logits = normalize(logits, 0.1, 0.9)
    return logits

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal((5, 5, 1, 6), mean = mu, stddev = sigma), name='conv1_W')
    conv1_b = tf.Variable(tf.zeros((6)), name='conv1_b')
    strides = [1, 1, 1, 1]
    conv1 = tf.nn.conv2d(x, conv1_W, strides=strides, padding='VALID', name='conv1') + conv1_b
    
    # Activation.
    conv1 = tf.nn.relu(conv1, name='conv1_actd')

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    strides = [1, 2, 2, 1]
    conv1 = tf.nn.max_pool(conv1, ksize=strides, strides=strides, padding='VALID', name='pool1')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal((5, 5, 6, 16), mean = mu, stddev = sigma), name='conv2_W')
    conv2_b = tf.Variable(tf.zeros((16)), name='conv2_b')
    strides = [1, 1, 1, 1]
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=strides, padding='VALID', name='conv2') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2, name='conv2_actd')

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    strides = [1, 2, 2, 1]
    conv2 = tf.nn.max_pool(conv2, ksize=strides, strides=strides, padding='VALID', name='pool2')

    # Flatten. Input = 5x5x16. Output = 400.
    flat1 = tf.contrib.layers.flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    flat2_W = tf.Variable(tf.truncated_normal((400, 120), mean = mu, stddev = sigma), name='flat2_W')
    flat2_b = tf.Variable(tf.zeros((120)), name='flat2_b')
    flat2 = tf.add(tf.matmul(flat1, flat2_W), flat2_b, name='flat2')
    
    # Activation.
    flat2 = tf.nn.relu(flat2, name='flat2_actd')

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    flat3_W = tf.Variable(tf.truncated_normal((120, 84), mean = mu, stddev = sigma), name='flat3_W')
    flat3_b = tf.Variable(tf.zeros((84)), name='flat3_b')
    flat3 = tf.add(tf.matmul(flat2, flat3_W), flat3_b, name='flat3')
    
    # Activation.
    flat3 = tf.nn.relu(flat3, name='flat3_actd')

    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    flat4_W = tf.Variable(tf.truncated_normal((84, n_classes), mean = mu, stddev = sigma), name='flat4_W')
    flat4_b = tf.Variable(tf.zeros((n_classes)), name='flat4_b')
    flat4 = tf.add(tf.matmul(flat3, flat4_W), flat4_b, name='output')
   
    logits = flat4
    return logits

EPOCHS = 2
BATCH_SIZE = 128
rate = 0.001

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)  
logits = preprocess(x)
logits = LeNet(x)
print("logits is:{}",logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y, name='xent')
loss_operation = tf.reduce_mean(cross_entropy,name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate = rate,name='opt')
training_operation = optimizer.minimize(loss_operation)



print("saver is pass")

def evaluate(X_data, y_data):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

saver = tf.train.Saver()
#*************************************************



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("num_examples = :{}".format(num_examples))
    
    print("Training...")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        print("start first cycle")
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            # print("batch_x is:{}",batch_x)
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            print("end first cycle")
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))