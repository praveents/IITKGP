# Name : Tadisetty Subbha Praveen
# Centre : Bangalore
# Specific compilation/execution flags (if required) : None

import pandas as pd
import numpy as np

# read the training data3
trainData = pd.read_csv('data3.csv', header=None)

# read the testing test3
test = pd.read_csv('test3.csv', header=None)

# find the unique classes
print(trainData.iloc[8].unique())

# find the classes values
num_classes = trainData.iloc[8].nunique()

print('number of classes in classification : ', num_classes)

# converting training input columns dataframe into ndarray
trainInput = trainData.iloc[:,0:8].values

# converting training output column dataframe into ndarray
trainOutput = trainData.iloc[:,8:9].values

# converting testing input columns dataframe into ndarray
testInput = test.iloc[:,0:8].values

class BernoulliNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        smoothing = 2 * self.alpha
        n_doc = np.array([len(i) + smoothing for i in separated])
        self.feature_prob_ = count / n_doc[np.newaxis].T
        return self

    def predict_log_proba(self, X):
        return [(np.log(self.feature_prob_) * x + \
                 np.log(1 - self.feature_prob_) * np.abs(x - 1)
                ).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)


nb = BernoulliNB(alpha=1).fit(trainInput, trainOutput)

testOutput = nb.predict(testInput)

id = 1

with open('tadisetty_praveen_3.out', 'w') as file:
    for i in list(testOutput):
        file.write("Test Instance %d: %d, " % (id, i))
        id += 1

