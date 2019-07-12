import numpy as np
import random
#missingno is a python library for nice visulaisation of missing number in the data
import missingno as msno

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x = np.array([1, 2, 3])
y = np.array([2, 1, 2])
a, b = genData(100, 25, 10)
print(np.shape(x))
m,n = np.shape(a)
m = 3
numIterations = 3
alpha = 0.1
theta = np.array([1.0, 0.0])
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)