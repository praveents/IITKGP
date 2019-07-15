import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

train_data = np.load('data_assg01/training_data.npy')
test_data = np.load('data_assg01/test_no_label.npy')
test_data = np.transpose(test_data)
train_data = np.transpose(train_data)
X = train_data[:, 0]
y = train_data[:, 1]
print('Training data shape:', train_data.shape)
print('Training data size:', train_data.size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('Training data shape:', X_train.shape)
print('Training data size:', X_train.size)
print('Training data shape:', X_test.shape)
print('Training data size:', X_test.size)

# create and fit the LSTM network
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal', input_dim=1, activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

NN_model.compile(loss='mean_squared_error', optimizer='adam')
NN_model.fit(X_train, y_train,  epochs=10, batch_size=32, verbose=2)

y_predict = NN_model.predict(X_test)

# rms = np.sqrt(np.mean(np.power((y_test - y_predict), 2)))

y_test_predict = NN_model.predict(test_data)

np.save("assgn1_1_test_labels_AI0004.npy", y_test_predict)

# print("rms:", rms)

print('Done')
