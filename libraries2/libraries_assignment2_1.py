# import all the required libraries
import tensorflow as tf
import numpy as np
import sys
# ================================================= QUESTION 1 =================================================

# Sample input numpy arrays ( Shape - N x 4 x 3 ) -
# Input Array  A =
#         [[[19, 15, 30],
#         [11, 23, 22],
#         [32, 32,  0],
#         [ 7,  1, 20]],
#     [[14, 18, 20],
#         [11, 12, 33],
#         [ 2, 35, 12],
#         [ 1, 24, 10]],
#     [[27,  6, 27],
#         [34, 21, 22],
#         [ 4,  8, 23],
#         [ 4, 18, 16]]]
# And
# Input Array B =
#             [[[16, 30,  3],
#             [ 9, 21, 27],
#             [32, 34,  1],
#             [35, 25, 26]],
#         [[14, 34, 24],
#             [ 1, 12, 25],
#             [18, 24, 25],
#             [27,  4, 21]]]

# Design a computation graph that takes as input a numpy array ( Similar to the sample array given ),
# and performs the following operations
# ( NOTE - Your computation graph can be provided either the A, or the B array as input. You should write code such that
# it can handle both of them.)

# *** NOTE - ALL THESE OPERATIONS SHOULD BE PERFORMED WITHOUT ANY EXPLICIT LOOPS. ****

# Normalization of each Batch - The input Data’s shape is N x 4 x 3. Assume that N is your batch size,
#                               so the input has N batches, each having 4 samples and 3 features.
#                               Normalize the features of each batch, i.e,
#                               Each feature needs to be subtracted by its mean, and divided by its stdev.
#                               Let the tensor after normalization be called - NORMALIZED_TENSOR. It’s shape would be the same as the input Data.
# Changing Shape - Given the input data having N batches,
#                  convert it into an tensor where the data of all batches is combined into one single batch.
#                  For example, if your input has shape N x 4 x 3, your output tensor should have shape N x 3.
#                  Let the output tensor be named   - BATCHES_COMBINED_TENSOR


# Matrices Sum - The input tensor consists of N batches, each being a 4 x 3 matrix.
#              Find the sum of all values in each batch ( each 4 x 3 Matrix ).
#              If your input has shape N x 4 x 3, output will have shape N. Let this output array be named - MATRICES_SUM.
# Softmax - Apply the softmax function on MATRICES_SUM. Let this output array be named - SOFTMAX_OUTPUT
# Maximum Value Index - Get the position of the largest element in the SOFTMAX_OUTPUT array.  Let this index be named - MAX_VALUE_INDEX

# print NORMALIZED_MATRIX, BATCHES_COMBINED_TENSOR,  MATRICES_SUM, and MAX_VALUE_INDEX

# *** NOTE - ALL THESE OPERATIONS SHOULD BE PERFORMED WITHOUT ANY EXPLICIT LOOPS. ****

tf.compat.v1.reset_default_graph()

g = tf.Graph()

# -------------------------------------------------------------------------------------------------------

#   YOUR CODE


vA = [[[19, 15, 30],
      [11, 23, 22],
      [32, 32, 0],
      [7, 1, 20]]]

A = [[[19, 15, 30],
      [11, 23, 22],
      [32, 32, 0],
      [7, 1, 20]],
     [[14, 18, 20],
      [11, 12, 33],
      [2, 35, 12],
      [1, 24, 10]],
     [[27, 6, 27],
      [34, 21, 22],
      [4, 8, 23],
      [4, 18, 16]]]

B = [[[16, 30, 3],
      [9, 21, 27],
      [32, 34, 1],
      [35, 25, 26]],
     [[14, 34, 24],
      [1, 12, 25],
      [18, 24, 25],
      [27, 4, 21]]]

def tensor_norm_manual(data):
    inp = tf.placeholder(shape=(None, 4, 3), dtype=tf.float32)
    std_dev = tf.math.reduce_std(inp, axis=1)
    mean = tf.math.reduce_mean(inp, axis=1)
    NORMALIZED_MATRIX = (inp - mean[:, None]) / (std_dev[:, None])
    BATCHES_COMBINED_TENSOR = tf.reshape(inp, [-1, tf.shape(inp)[-1]])
    MATRICES_SUM = tf.math.reduce_sum(inp, axis=(-1, -2))
    SOFTMAX_OUTPUT = tf.nn.softmax(MATRICES_SUM)
    MAX_VALUE_INDEX = tf.argmax(SOFTMAX_OUTPUT)
    with tf.Session() as sess:
        input_tensor = data
        # np.array(
        #     [[[10, 4, 2], [1, 2, 22], [2, 42, 0], [7, 1, 10]], [[14, 12, 20], [11, 12, 33], [2, 15, 12], [1, 24, 10]],
        #      [[27, 6, 27], [34, 21, 22], [4, 8, 23], [4, 18, 16]]])
        normalized_matrix, batches_combined_tensor, matrices_sum, max_value_index = sess.run(
            [NORMALIZED_MATRIX, BATCHES_COMBINED_TENSOR, MATRICES_SUM, MAX_VALUE_INDEX], feed_dict={inp: input_tensor})
        print(normalized_matrix)
        print(batches_combined_tensor)
        print(matrices_sum)
        print(max_value_index)
    return

def tensor_norm(inputX):
    # input features and labels

    with tf.compat.v1.Session() as sess:

        tf.compat.v1.global_variables_initializer()

        inputTensor = tf.compat.v1.placeholder(name="inputTensor", shape=[None, 4, 3], dtype=np.float32)

        # meanTensor = tf.compat.v1.placeholder(name="meanTensor", shape=[None, 3], dtype=np.float32)
        #
        # varianceTensor = tf.compat.v1.placeholder(name="varianceTensor", shape=[None, 3], dtype=np.float32)



        sess.run(tf.global_variables_initializer())

        epsilon = 1e-10

        # convert the input to a tensor type
        convTensor = tf.convert_to_tensor(inputX, dtype=tf.float32)

        # reshape the input placeholder to the input tensor shape
        inputTensor = tf.reshape(inputTensor, convTensor.shape)

        # feed the placeholder with the input tensor
        sess.run(inputTensor, feed_dict={inputTensor: inputX})

        axis = list(range(len(convTensor.shape)-1))

        meanTensor = tf.math.reduce_mean(convTensor, axis=1)

        varianceTensor = tf.math.reduce_variance(convTensor, axis=1)

        outputTensor = tf.nn.batch_normalization(inputTensor, meanTensor, varianceTensor,
                                            offset=None, scale=None,
                                            variance_epsilon=1e-3)

        result = sess.run(meanTensor)

        print("Mean :", result )

        print("Variance :", sess.run(varianceTensor))

        result = sess.run(outputTensor, feed_dict={inputTensor: inputX})

        # run normalization function, pass input tensor, median and variance
        print("Normalized :", result)

        # print the normalized tensor and check the shape, should be same as input
        print('TF Normalized Tensor:', outputTensor)



tensor_norm(vA)
tensor_norm_manual(vA)
tensor_norm(A)
tensor_norm_manual(A)
tensor_norm(B)
tensor_norm_manual(B)
# -------------------------------------------------------------------------------------------------------
#########################################################################################################


############################################  QUESTION 2  ###############################################

# Create a computation graph of function : ( exp^(f) - exp^(-f) )/( exp^(f) + exp^(-f) ) + (1 / ( 1 + exp^(-f) ))
# Here,  f = W*X  + b.  W, X and b are defined below -
# W - 2x2 matrix from any normal distribution
# X - Initialize with value = [2, 4]
# b - Array of all ones
# (i) Define one graph where W, X and b are constants values in the graph.
# (ii) Define another graph where W, X, and b are passed as inputs to the graph.

# 1. What is the difference between (i) and (ii)?
# 2. Did you use tf.variable in (i) or (ii) ? Why or why not?
# Write the code in the YOUR CODE section.

tf.compat.v1.reset_default_graph()


# -------------------------------------------------------------------------------------------------------

#   YOUR CODE

def linreg_const():
    W = tf.compat.v1.random.normal(shape=[2, 2], seed=32)
    X = tf.constant([2, 4], dtype=tf.float32)
    b = tf.ones(shape=[1], dtype=tf.float32)
    f = tf.add(tf.multiply(X, W), b)
    result = (tf.exp(f) - tf.exp(-f)) / (tf.exp(f) + tf.exp(-f)) + (1 / (1 + tf.exp(-f)))

    with tf.compat.v1.Session() as sess:
        print("Constant input W :", sess.run(W))
        print("Constant input Result :", sess.run(result))


def linreg_var(W, X, b):
    f = tf.add(tf.multiply(X, W), b)
    result = (tf.exp(f) - tf.exp(-f)) / (tf.exp(f) + tf.exp(-f)) + (1 / (1 + tf.exp(-f)))

    with tf.compat.v1.Session() as sess:
        print("Variable input W :", sess.run(W))
        print("Variable input Result :", sess.run(result))


linreg_const()

vW = tf.compat.v1.random.normal(shape=[2, 2])
vX = tf.constant(value=[1, 2], dtype=tf.float32)
vb = tf.ones(shape=[1], dtype=tf.float32)

linreg_var(vW, vX, vb)

# Your answer to the questions -
# -------------------------------------------------------------------------------------------------------
#########################################################################################################

############################################  QUESTION 3  ###############################################

# Create function named ‘mean_var’ which takes input constant integer 'x' and returns two tensors
# a) mean - tensors of shape 'x' from any normal distribution
# b) var - tensors of shape 'x' from any normal distribution
# Randomly initialize the graph and save this as ‘mean_var.ckpt’

# Now reset the graph using tf.reset_default_graph()

# Create another function named ‘ normal_sample’ which takes as input two tensors ‘mean’ and ‘var’  and alpha. This
# function returns a vector of dimension ‘x’ which is equal to mean + alpha*var where alpha is constant.

# Load the graph from the mean_var.ckpt and pass a value of x to function ‘mean_var’ as 300. Get the mean and var and
# pass it to the function ‘normal_sample’ and alpha as 0.1.
# Get the output of the function and print it.

tf.compat.v1.reset_default_graph()

# -------------------------------------------------------------------------------------------------------

#   YOUR CODE

# -------------------------------------------------------------------------------------------------------
#########################################################################################################

############################################  QUESTION 4  ###############################################

# Consider the following Computation Graph -

# ----------------------------------------- COMPUTATION GRAPH --------------------------------------------

"""

import tensorflow as tf
import numpy as np
input = tf.placeholder(shape=(1, 224, 224, 3), dtype=tf.float32)
output = input + 5
with tf.Session() as sess:
    network_input = np.random.randint(5,  size=(700, 350, 3))
    out = sess.run(output, feed_dict = {input : network_input})

"""

# ----------------------------------------- COMPUTATION GRAPH --------------------------------------------
# Will this code run without any errors? If not, what changes have to be made here, and why?
# Write the modified code below ( in the YOUR CODE section )

tf.compat.v1.reset_default_graph()

import tensorflow as tf
import numpy as np

input = tf.placeholder(shape=(1, 224, 224, 3), dtype=tf.float32)
output = input + 5
with tf.Session() as sess:
    network_input = np.random.randint(5, size=(1, 224, 224, 3))
    network_input = np.resize(network_input, (1, 224, 224, 3))
    out = sess.run(output, feed_dict={input: network_input})

# -------------------------------------------------------------------------------------------------------

#   YOUR CODE

# Your reasoning / answer here -
# -------------------------------------------------------------------------------------------------------
#########################################################################################################
############################################ QUESTION 5 ############################################################
# Train a classification model for  three classes/diseases: Bacterial leaf blight, Brown spot, and Leaf smut,
# each having 40 images. Image Format: .jpg, The images were captured with a white background, in direct sunlight.
# The images were reduced to the desired resolution for processing.
#
# Write the script to train from the given dataset. You can download the dataset from e below link
# link : https://archive.ics.uci.edu/ml/machine-learning-databases/00486/rice_leaf_diseases.zip
#####################################################################################################################

import os, glob
import time

import tensorflow as tf
import keras
import numpy as np

num_features = 897
N_CLASSES = 3
path_of_dataset = './rice_leaf_diseases'  # path to your dataset


# Step 1: Read in data
# -------------------------------------------------------------------------------------------------------
def preprocess_image(image):
    """
        Reads and outputs the resized and normalized form of the input filename
        INPUT : A Tensor of type string.
        OUTPUT : Resized and normalized Image
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [64, 64])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    """
        Reads and outputs the entire contents of the input filename
        INPUT : Image path of type string.
        OUTPUT : Resized and normalized Image
    """
    image = tf.read_file(path)
    return preprocess_image(image)


# Use these functions for loading the image while training the dataset

# -------------------------------------------------------------------------------------------------------

# Step 2: Define paramaters for the model < Can be changed as per convenience >
LEARNING_RATE = 0.001
BATCH_SIZE = 16
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 10

# Step 3: create placeholders for features and labels
# each image in the  data is of shape < find this shape >
# therefore, each image is represented with a < above shape > tensor
# there are 3 classes for each image,
# Features are of the type float, and labels are of the type int
# -------------------------------------------------------------------------------------------------------
#   YOUR CODE
# -------------------------------------------------------------------------------------------------------
# initialize a tensorflow graph
graph = tf.Graph()

tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, num_features])
tf_train_labels = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, N_CLASSES])

# Variables.
weights = tf.Variable(tf.truncated_normal([num_features, N_CLASSES]))
biases = tf.Variable(tf.zeros([N_CLASSES]))

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability label of the image
# -------------------------------------------------------------------------------------------------------
#   YOUR CODE
# -------------------------------------------------------------------------------------------------------

logits = tf.matmul(tf_train_dataset, weights) + biases

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
# -------------------------------------------------------------------------------------------------------
#   YOUR CODE
# -------------------------------------------------------------------------------------------------------

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=tf_train_labels, logits=logits))

# Step 6: define training op
# using gradient descent to minimize loss
# save the model after every 5 steps
# -------------------------------------------------------------------------------------------------------
#   YOUR CODE
# -------------------------------------------------------------------------------------------------------

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)


# Step 7: create a session and run the training
# minimize the training op using sess.run
# also print the loss at every step
# -------------------------------------------------------------------------------------------------------
#   YOUR CODE
# -------------------------------------------------------------------------------------------------------
# utility function to calculate accuracy

def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu


with tf.Session(graph=graph) as session:
    # initialize weights and biases
    tf.global_variables_initializer().run()
    print("Initialized")

    for step in range(N_EPOCHS):
        # pick a randomized offset
        offset = np.random.randint(0, train_labels.shape[0] - BATCH_SIZE - 1)

        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

        # Prepare the feed dict
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels}

        # run one step of computation
        _, l, predictions = session.run([optimizer, loss, train_prediction],
                                        feed_dict=feed_dict)

        if (step % 500 == 0):
            print("Minibatch loss at step {0}: {1}".format(step, l))
            print("Minibatch accuracy: {:.1f}%".format(
                accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}%".format(
                accuracy(valid_prediction.eval(), valid_labels)))

    print("\nTest accuracy: {:.1f}%".format(
        accuracy(test_prediction.eval(), test_labels)))
