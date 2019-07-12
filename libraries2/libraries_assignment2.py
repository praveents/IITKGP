##1 What is the output?
import tensorflow as tf
import numpy as np
x = tf.constant(32, dtype='float32')
y = tf.Variable(x*2 + 11)
model = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(model)
    print(sess.run(y))

'''' prints 75.00'''

##2. Create one constant, a that is 10 and one variable that double of a
# Write code in TensorFlow
a = tf.constant(10, dtype='float32')

da = tf.Variable(a*2, dtype='float32')

model = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(model)
    print(sess.run(da))

#3 Why do we need placeholder?
''' Place holder is a variable that will get assigned with data later.
It will allow to create graph with operations without needing the data.
Place holders are initially empty and are used to feed in the actual training data.
They do need to declare expected data type with optional share argument.'''

# 4 y=Wx+b is the model design. For building this model, which ones should be variables and why?
''' W and b are the variables which holds the initial slope and intercept. These variables can
be used as input to operations in the graph. Variables has to be explicity initialized before 
using in the operations. We need to use variables for W and b as the values has to changed as the model 
converges. Constants doesn't allow to change the values'''
W = tf.Variable(0.1, name="W")
b = tf.Variable(0.01, name="b")

x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)


#5 Find the error and why
x = tf.placeholder("float", [2, 4])
y = x * 10 + 1
with tf.Session() as session:
    dataX = [[12, 2, 0, -2],
              [14, 4, 1, 0],]
    placeX = session.run(y, feed_dict={x: dataX})
    print(placeX)

''' Error : Cannot feed data of shape 2*4 in to place holder x of shape 3*4.'''

#6 #Load CIFAR-10 Data and tell me shape of data
from keras.datasets import cifar10
np.random.seed(100) # for reproducibility
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("Shape of X train:", X_train.shape)
print("Shape of X train:", X_test.shape)


