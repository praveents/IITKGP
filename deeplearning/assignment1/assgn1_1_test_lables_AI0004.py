import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import keras
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.advanced_activations import PReLU

print(mpl.rcParams['agg.path.chunksize'])
mpl.rcParams['agg.path.chunksize'] = 20000

train_data = np.load('data_assg01/training_data.npy')
test_data = np.load('data_assg01/test_no_label.npy')
test_data = np.transpose(test_data)
train_data = np.transpose(train_data)

train_df = pd.DataFrame(train_data)

print(train_df.describe())

X = train_data[:, 0]
y = train_data[:, 1]

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

transformer = Normalizer().fit(train_df)  # fit does nothing.
train_df = transformer.transform(train_df)



# transformer = Normalizer().fit(X)  # fit does nothing.
# X = transformer.transform(X)
# transformer = Normalizer().fit(y)  # fit does nothing.
# y = transformer.transform(y)
# transformer = Normalizer().fit(test_data)  # fit does nothing.
# test_data = transformer.transform(test_data)

# X = train_df[:, 0]
# y = train_df[:, 1]

print('Actual data shape:', train_data.shape)
print('Actual data size:', train_data.size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

minmax_scalar = MinMaxScaler(feature_range=(-1, 1))
X_train = minmax_scalar.fit_transform(X_train)
X_test = minmax_scalar.fit_transform(X_test)
y_train = minmax_scalar.fit_transform(y_train)
y_test = minmax_scalar.fit_transform(y_test)
test_data = minmax_scalar.fit_transform(test_data)

plt.hist(X_train, bins='auto', color='r')
plt.savefig('xtrain.png')
plt.close()
plt.hist(X_test, bins='auto', color='g')
plt.savefig('xtest.png')
plt.close()
plt.hist(y_train, bins='auto', color='b')
plt.savefig('ytrain.png')
plt.close()
plt.hist(y_test, bins='auto', color='y')
plt.savefig('ytest.png')
plt.close()
print('Training data shape:', X_train.shape)
print('Training data size:', X_train.size)
print('Training data shape:', X_test.shape)
print('Training data size:', X_test.size)

activation_function = 'relu'
# create and fit the network
NN_model = Sequential()


# The Input Layer :
NN_model.add(Dense(512, kernel_initializer='normal', input_dim=1,activation=activation_function))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal', activation=activation_function))
NN_model.add(Dense(128, kernel_initializer='normal', activation=activation_function))
NN_model.add(Dense(64, kernel_initializer='normal', activation=activation_function))
NN_model.add(Dense(32, kernel_initializer='normal', activation=activation_function))
NN_model.add(Dense(16, kernel_initializer='normal', activation=activation_function))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal'))

# Compile the network :
#  keras.optimizers.SGD(lr=0.001)
NN_model.compile(loss='mean_squared_error', optimizer= keras.optimizers.Adadelta() , metrics=['mean_squared_error'])
NN_model.summary()

# NN_model.compile(loss='mean_squared_error', optimizer='adam')
NN_model.fit(X_train, y_train,  epochs=4, batch_size=32, verbose=2)

y_predict = NN_model.predict(X_test)
plt.hist(y_predict, bins='auto', color='b')
plt.savefig(activation_function+'_ytest_predict.png')
plt.close()

y_test_predict = NN_model.predict(test_data)

plt.hist(y_test_predict, bins='auto', color='skyblue')
plt.savefig(activation_function+'_testdata_predict.png')
plt.close()
np.save("assgn1_1_test_labels_AI0004.npy", y_test_predict)

# print("rms:", rms)

print('Done')
