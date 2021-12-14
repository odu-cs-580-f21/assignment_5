import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./train.csv')
data = np.array(data)

# m = rows
# n = columns
m, n = data.shape

# shuffle before splitting into dev and training sets
np.random.shuffle(data) 

# splitting our data into training and dev sets
# transposing our data to turn our rows into columns
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

# what is not used for dev set, is used for training
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def initParams():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# ReLU(x) linear function
# x if x >= 0
# 0 if x <= 0
def ReLU(Z):
    return np.maximum(Z, 0)

# softmax activation function
# softmax(x) = e^z / sum(e^z)
# takes in our output layer Z and returns our probabilities
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forwardProp(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# converts bool to 1 or 0 and returns the slope derivative
def ReLUDeriv(Z):
    return Z > 0

def oneHot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backwardProp(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = oneHot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLUDeriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def getPredictions(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradientDescent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = initParams()
    # for every iteration desired by the function argument
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwardProp(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # every 10th iteration, print accuracy
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = getPredictions(A2)
            print(getAccuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradientDescent(X_train, Y_train, 0.5, 100)

def makePredictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardProp(W1, b1, W2, b2, X)
    predictions = getPredictions(A2)
    return predictions

def testPrediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = makePredictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

testPrediction(4, W1, b1, W2, b2)