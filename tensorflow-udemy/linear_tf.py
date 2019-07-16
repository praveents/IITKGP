import tensorflow as tf
import numpy as np

# set random seed to have same
np.random.seed(28)
tf.set_random_seed(28)

w = tf.Variable(tf.random_normal([10, 3]))
b = tf.Variable(tf.ones([3]))
x = tf.placeholder(tf.float32, (None, 10))
x_data = np.random.uniform(0, 10, (5, 5)) + np.random.uniform(-1, 1, (5, 5))
y_data = np.random.uniform(0, 10, (5, 5)) + np.random.uniform(-1, 1, (5, 5))
a = tf.placeholder(tf.float32)
xW = tf.matmul(x, w)
z = tf.add(xW, b)
a = tf.sigmoid(z)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x: np.random.random([1, 10])})

print(layer_out)
