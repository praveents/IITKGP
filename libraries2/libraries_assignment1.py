from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import csv

# read the train data
taiwan_data = pd.read_csv('train.csv')

#
predict_data = pd.read_csv('test.csv')

# create input dataframe
customer_data_train = taiwan_data.copy()

customer_features = customer_data_train.columns

# drop last column (default payment)
customer_data_train.drop(customer_features[-1], axis=1, inplace=True)

# drop ID and  unnamed columns
customer_data_train.drop(['Unnamed: 0', 'ID', ], axis=1, inplace=True)
predict_data.drop(['Unnamed: 0', 'ID', ], axis=1, inplace=True)

# crete
default_payment_train = taiwan_data.copy()

default_payment_train.drop(customer_features[:-1], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(customer_data_train, default_payment_train, test_size=0.2, random_state=42)
y_train  = y_train.astype(int)
y_test  = y_test.astype(int)
batch_size =len(X_train)

print(X_train.shape, y_train.shape, y_test.shape )

## resclae

scaler = MinMaxScaler()
# Train
X_train_scaled = scaler.fit_transform(X_train.astype(np.int))
# test
X_test_scaled = scaler.fit_transform(X_test.astype(np.int))

# test
X_predict_scaled = scaler.fit_transform(predict_data.astype(np.int))


ann = MLPClassifier()
ann_parameters = {'hidden_layer_sizes': [(100,1), (100,2), (100,3)],
                  'alpha': [.0001, .001, .01, .1, 1],
                 }

acc_scorer = make_scorer(f1_score)

# Run the 10-fold grid search
grid_obj = GridSearchCV(ann, ann_parameters, cv = 5, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train_scaled, y_train)

# Set the algorithm to the best combination of parameters
ann = grid_obj.best_estimator_

# Fit the best algorithm to the data.
ann.fit(X_train_scaled, y_train)
ann.predict(X_test_scaled)

print(round(ann.score(X_test_scaled, y_test) * 100, 2))

y_predict = ann.predict(X_predict_scaled)

with open('returns.csv', 'w') as f:
    writer = csv.writer(f)
    for val in y_predict:
        writer.writerow([val])


