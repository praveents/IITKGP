import numpy as np
import tensorflow as tf

# Question 1
# Create a tensor of shape (2,3)  having normal distribution having mean 4 and std 0.5
# Create a tensor Y of shape (2,3) having a sequence of numbers 12,15,18,...., 150
# Get and Print sum of these two Tensors that is Z=  X+Y   (Shape of Z would be same as that of X or Y)

tf.set_random_seed(23)

X = tf.random.normal((2,3), mean=4, stddev=0.5, dtype=tf.float32)
y_seq = np.resize(np.arange(12, 153, 3), (2, 3))
Y = tf.constant(value=y_seq, dtype=tf.float32)
Z = tf.add(X, Y)

with tf.Session() as sess:
    print(sess.run(Z))

# Question 2
#Get a new tensor W = tanh(Z)
#Get a new tensor T= sigmoid(Z)

w = tf.tanh(Z)
T = tf.sigmoid(Z)

with tf.Session() as sess:
    print(sess.run(w))
    print(sess.run(T))


## Question 3
# Replacing sigmoid/tanh with another(new) activation function that is ( exp^(f) - exp^(-f) )/( exp^(f) + exp^(-f) ) + (1 / ( 1 + exp^(-f) ))
import tensorflow as tf
W = tf.random_normal(shape=(3,2))
X = tf.constant([2,4], dtype=tf.float32)
B = tf.zeros(shape=(2))
f = W*X + B
newActivationResult = tf.sigmoid(f) #((tf.exp(f) - tf.exp(-f)) / (tf.exp(f) + tf.exp(-f))) + (1 / ( 1 + tf.exp(-f)))
with tf.Session() as sess:
    output = sess.run(newActivationResult)
    print(output)


#If we want to replace new activation function with sigmoid, which line is to be changed?

# Question 4
import tensorflow as tf
import numpy as np
inp = tf.placeholder(shape=(1, 32, 32, 3), dtype=tf.float32)
output = 2*inp + 5
with tf.Session() as sess:
    network_input = np.random.randint(5,  size=(256, 3))
    network_input = np.resize(network_input, (1, 32,32, 3))
    out = sess.run(output, feed_dict = {inp: network_input})
    print(out)


# Question 5
# Build an image(digit) classifier using keras for MNIST data using adagrad, 'binary_crossentropy' using MLP (not CNN)

from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('Training data shape : ', train_images.shape, train_labels.shape)

print('Testing data shape : ', test_images.shape, test_labels.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[10, 5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_images[0, :, :], cmap='gray')
plt.title("Ground Truth : {}".format(train_labels[0]))
plt.show()

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_images[0, :, :], cmap='gray')
plt.title("Ground Truth : {}".format(test_labels[0]))
plt.show()

# Change from matrix to array of dimension 28x28 to array of dimention 784
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# Change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
print('Original label 0 : ', train_labels[0])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(nClasses, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Plot the Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()

# Plot the Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()

from keras.layers import Dropout

model_reg = Sequential()
model_reg.add(Dense(512, activation='relu', input_shape=(dimData,)))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(512, activation='relu'))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(nClasses, activation='softmax'))

model_reg.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history_reg = model_reg.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                            validation_data=(test_data, test_labels_one_hot))

# Plot the Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history_reg.history['loss'], 'r', linewidth=3.0)
plt.plot(history_reg.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()

# Plot the Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history_reg.history['acc'], 'r', linewidth=3.0)
plt.plot(history_reg.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()

# Predict the most likely class
model_reg.predict_classes(test_data[[0],:])

# Predict the probabilities for each class
model_reg.predict(test_data[[0],:])