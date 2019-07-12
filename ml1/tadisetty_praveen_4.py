# Name : Tadisetty Subbha Praveen
# Centre : Bangalore
# Specific compilation/execution flags (if required) : None

import math
import operator
import pandas as pd

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# read the training data4
trainData = pd.read_csv('data4.csv', header=None)

# read the testing test4
test = pd.read_csv('test4.csv', header=None)

# find the unique classes
print(trainData.iloc[8].unique())

# find the classes values
num_classes = trainData.iloc[8].nunique()

print('number of classes in classification : ', num_classes)

# converting training input columns dataframe into ndarray
trainInput = trainData.iloc[:, :].values

# converting testing input columns dataframe into ndarray
testInput = test.iloc[:, :].values
# testInput = trainData.iloc[16:, :].values

# generate predictions
predictions = []
k = 5
for x in range(len(testInput)):
    neighbors = getNeighbors(trainInput, testInput[x], k)
    result = getResponse(neighbors)
    predictions.append(result)

id = 1

with open('tadisetty_praveen_4.out', 'w') as file:
    for i in predictions:
        file.write("Test Instance %d: %d, " % (id, i))
        id += 1


