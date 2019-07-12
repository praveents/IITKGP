##################################################
## 1. Load breast cancer dataset from sklearn
## 2. Split the data as 70:30 as train and test data
## 3. Fit the train data into SVM model with diffferent kernels
##    and bar plot the accuracy of different SVM model with the test data
## 4. Fit the above training dataset into a SVM model with ploynomial kernel
##    with varying degree and plot the accuracy wrt. degree of ploynomial kernel with the test data
## 5. Define a custom kernel K(X,Y)=K*XY'+theta where k and theta are constants
## 6. Use the custom kernel and report the accuracy with the given train and test dataset
##################################################

##################################################
## Basic imports
## You are not required to import additional module imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import time
from sklearn.preprocessing import PolynomialFeatures
###################################################

###################################################
## the method loads breast cancer dataset and returns
## the dataset and label as X,Y
def load_data():
	data_set = datasets.load_breast_cancer()
	X=data_set.data
	y=data_set.target
	return X,y
###################################################

###################################################
## this method takes train and test data and different 
## svm models and fit the train data into svm models and 
## do bar plot using sns.barplot() of different svm model 
## accuracy. You need to implement the model fitting and 
## bar plotting in this method.
def svm_models(X_train, X_test, y_train, y_test,models):
	## write your own code here
	train_acc_arr = []
	# create dataframe to plot model, test and accuracy
	result_dataframe = pd.DataFrame(columns=['model', 'accuracy', 'test'])

	# fill predefined columns
	result_dataframe['model'] = ['linear', 'linear', 'rbf', 'rbf', 'poly', 'poly']
	result_dataframe['test'] = ['train', 'test', 'train', 'test', 'train', 'test']

	# for each model fit the model and get the scores
	for model in models:

		svm_classifier = model.fit(X_train, y_train)

		# get scores for train and test
		train_accuracy = svm_classifier.score(X_train, y_train)
		test_accuracy = svm_classifier.score(X_test, y_test)

		# append train and test scores
		train_acc_arr.append(train_accuracy)
		train_acc_arr.append(test_accuracy)

	# update df with the score list
	result_dataframe['accuracy'] = train_acc_arr

	# plot the bar plot for each model, train and test
	sns.barplot(x="model", y="accuracy", hue="test", data=result_dataframe)

	#display the plot
	plt.show()


###################################################

###################################################
## this method fits the dataset to a svm model with 
## polynomial kernel with degree varies from 1 to 3 
## and plots the execution time wrt. degree of 
## polynomial, you can calculate the elapsed time 
## by time.time() method
def ploy_kernel_var_deg(X_train, X_test, y_train, y_test):
	## write your own code here
	result_dataframe = pd.DataFrame(columns=['degree', 'time'])

	result_dataframe['degree'] = [1, 2, 3]

	# normalize the data before model fitting
	scaling = MinMaxScaler(feature_range=(0, 1)).fit(X_train)

	# transform X-train input
	X_train = scaling.transform(X_train)

	# list to hold execution times
	executionTime = []

	for i in range(1, 4):
		model = svm.SVC(C=1.0, kernel='poly', degree=i, gamma=2)

		start_time = time.time()
		model.fit(X_train, y_train)
		end_time = time.time()

		# update the execution time
		executionTime.append(end_time-start_time)

	result_dataframe['time'] = executionTime

	# plot the bar graph for execution times for each degree
	sns.barplot(x="degree", y="time", data=result_dataframe)
	plt.show()

###################################################

###################################################
## this method implements a custom kernel technique 
## which is K(X,Y)=k*XY'+theta where k and theta are
## constants. Since SVC supports custom kernel function
## with only 2 parameters we return the custom kernel 
## function name from another method which takes k and
## theta as input
def custom_kernel(k,theta):

	def my_kernel(X, Y):
		## write your own code here
		return ((k * np.dot(X, Y.transpose())) + theta)
	## write your own code here
	return my_kernel
####################################################

####################################################
## this method uses the custom kernel and fit the 
## training data and reports accuracy on test data
def svm_custom_kernel(X_train, X_test, y_train, y_test, model):
	## write your code here
	clf = model.fit(X_train, y_train)
	train_acc = clf.score(X_train, y_train)
	test_acc = clf.score(X_test, y_test)
	print('custom svm kernel train accuracy :', train_acc)
	print('custom svm kernel test accuracy :', test_acc)

####################################################

####################################################
## main method:
def main():
	X,y=load_data()
	# Split dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test

	C=1
	models = (svm.SVC(kernel='linear', C=C),
			  svm.SVC(kernel='rbf', gamma='auto', C=C),
			  svm.SVC(kernel='poly', degree=2, gamma='auto', C=C))

	svm_models(X_train, X_test, y_train, y_test,models)

	ploy_kernel_var_deg(X_train, X_test, y_train, y_test)

	k=0.1

	theta=0.1

	model=svm.SVC(kernel=custom_kernel(k,theta))
	svm_custom_kernel(X_train, X_test, y_train, y_test, model)
#####################################################	


if __name__=='__main__':
	main()



	
