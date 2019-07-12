#This code generates random data from Uniform Distribution and assigns labels.
#The data is a non-linear one with points inside a circle of fixed radius marked as -1 and outside as +1.
#We flip the labels of some data (here with 5% probability) to introduce some noise.
#You will be using Decision Tree and Naive Bayes Classifiers to classify the above generated data.


import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
#Do all the necessary imports here

def generate_data():

	np.random.seed(123) #Set seed for reproducibility. Please do not change/remove this line.
	x = np.random.uniform(-1,1,(128,2)) #You may change the number of samples you wish to generate
	y=[]
	for i in range(x.shape[0]):
		y.append(np.sign(x[i][0]**2 + x[i][1]**2 - 0.5)) #Forming labels
	return x,y

def flip_labels(y):

	num = int(0.05 * len(y)) #5% of data to be flipped
	np.random.seed(123)
	changeind = np.random.choice(len(y),num,replace=False) #Sampling without replacement
	#For example, np.random.choice(5,3) = ([0,2,3]); first argument is the limit till which we intend to pick up elements, second is the number of elements to be sampled

	#Creating a copy of the array to modify
	yc=np.copy(y) # yc=y is a bad practice since it points to the same location and changing y or yc would change the other which won't be desired always
	#Flip labels -1 --> 1 and 1 --> -1
	for i in changeind:
		if yc[i]==-1.0:
			yc[i]=1.0
		else:
			yc[i]=-1.0

	return yc

#Fill up the below function
def train_test_dt(x,y):
	# Perform a k-fold cross validation using Decision Tree
	#Plot train and test accuracy with varying k (1<=k<=10)
	avg_train_accuracy = []
	avg_test_accuracy = []
	decesionTree = DecisionTreeClassifier(max_leaf_nodes=5)
	for i in range(2, 10):
		kf = KFold(n_splits=i)
		kf.get_n_splits(x)
		train_accuracy = []
		test_accuracy = []
		for train_index, test_index in kf.split(x):
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]
			decesionTree.fit(x_train, y_train)
			train_accuracy.append(decesionTree.score(x_train, y_train))
			test_accuracy.append(decesionTree.score(x_test, y_test))
		# print('Accuracy of training set:', sum(train_accuracy)/len(train_accuracy))
		# print('Accuracy of testing set:', sum(test_accuracy)/len(test_accuracy))
		avg_train_accuracy.append(sum(train_accuracy)/len(train_accuracy))
		avg_test_accuracy.append(sum(test_accuracy) / len(test_accuracy))
	print('Decision Tree Accuracy of training set:', avg_train_accuracy)
	print('Decision Tree Accuracy of testing set:', avg_test_accuracy)
	plt.plot(np.arange(2,10), avg_train_accuracy, label='train accuracy')
	plt.plot(np.arange(2,10), avg_test_accuracy, label='test accuracy')
	plt.xticks(np.arange(1, 11, step=1))
	plt.yticks(np.arange(0, 1.5, step=0.1))
	plt.title('Decision Tree Classifier')
	plt.legend()
	plt.show()

#Fill up the velow function
def train_test_nb(x,y):

	# Perform a k-fold cross validation using Naive Bayes
	#Plot train and test accuracy with varying k (1<=k<=10)
	avg_train_accuracy = []
	avg_test_accuracy = []
	# Create a Gaussian Classifier
	gaussianNB = GaussianNB()
	for i in range(2, 10):
		kf = KFold(n_splits=i)
		kf.get_n_splits(x)
		train_accuracy = []
		test_accuracy = []
		for train_index, test_index in kf.split(x):
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]
			gaussianNB.fit(x_train, y_train)
			train_accuracy.append(gaussianNB.score(x_train, y_train))
			test_accuracy.append(gaussianNB.score(x_test, y_test))
		# print('Accuracy of training set:', sum(train_accuracy)/len(train_accuracy))
		# print('Accuracy of testing set:', sum(test_accuracy)/len(test_accuracy))
		avg_train_accuracy.append(sum(train_accuracy)/len(train_accuracy))
		avg_test_accuracy.append(sum(test_accuracy) / len(test_accuracy))
	print('Gaussian NB Accuracy of training set:', avg_train_accuracy)
	print('Gaussian NB Accuracy of testing set:', avg_test_accuracy)
	plt.plot(np.arange(2,10), avg_train_accuracy, label='train accuracy')
	plt.plot(np.arange(2,10), avg_test_accuracy, label='test accuracy')
	plt.xticks(np.arange(1, 11, step=1))
	plt.yticks(np.arange(0, 1.5, step=0.1))
	plt.title('Naive Bayes Classifier')
	plt.legend()
	plt.show()



def main():

	x,y = generate_data() #Generate data
	y = flip_labels(y) #Flip labels
	y=np.asarray(y) #Change list to array
	train_test_dt(x,y)
	train_test_nb(x,y)


if __name__=='__main__':
	main()
