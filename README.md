# CS 480/580 - Assignment_5 - Dániel B. Papp

# Notice

Since I wasn't able to use the dataset provided in the assignment document, I used the same dataset but provided by Kaggle in a `.csv` format. The data and the Kaggle challenge can be found [here](https://www.kaggle.com/c/digit-recognizer/data).

# How to run

```console
python3 app.py
```

# How it works

On a high level, the program reads the `.csv` dataset and makes predictions for each image. The predictions are then compared to the actual labels and the accuracy is calculated. The accuracy is then printed to the terminal.

All the necessary mathematical operations will be explained function by function below.

# Neural Network (NN) structure overview

Our NN will have three layers in total. To further improve the accuracy of our model, we could increase the number or layers or the number of nodes in the hidden layer.

1. Input layer
   - The input layer has 784 nodes, which corresponds with the 28x28 pixels of the image. Each individual pixel is normalized to a value between 0 and 1. Since we only care about the pixel either being `rgb(0,0,0)` or `rgb(255,255,255)`, we can normalize each pixel to represent true or false (1 or 0) weather it is black or white.
2. Hidden layer
   - For the sake of simplicity, we will use a single hidden layer with 10 nodes. The value of each node is calculated based on weights and biases applied to the value of the 784 nodes in the input layer.
3. Output layer
   - The output layer has 10 nodes, each representing a digit from 0 through 9. The value of each node is calculated based on weights and biases applied to the value of the 10 nodes in the hidden layer using a softmax activation algorithm.

# The mathematical operations and functions

## Our data

Each of our training example can be represented as a vector of 784 values. These vectors are then stacked into a matrix so we can calculate error from all examples at once with matrix operations.

Our matrix will have an `m x n` dimension, where m is the number of training examples and n is the number of nodes in the input layer (784). We transpose the matrix so the dimensions will be `n x m`, with each column corresponding to a training example and each row is a node.

## Weight and biases

Between every two layers is a set of connections between every node in the previous layer and every node in the following layer. Which means there is a weight of of ![img](http://www.sciweavers.org/tex2img.php?eq=w_%7Bi%2Cj%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) for every `i` in the number of nodes in the previous layer and every `j` in the number of nodes in the following layer.

From this we can conclude that our weights as a matrix will ![img](http://www.sciweavers.org/tex2img.php?eq=n%5E%7B%5Bl%5D%7D%20%20%5Ctimes%20n%5E%7B%5Bl%20-%201%5D%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) where ![img](http://www.sciweavers.org/tex2img.php?eq=n%5E%7B%5Bl%20-%201%5D%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) is the number of nodes in the previous layer and ![img](http://www.sciweavers.org/tex2img.php?eq=n%5E%7B%5Bl%5D%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) is the number of nodes in the following layer. We call this matrix the ![img](http://www.sciweavers.org/tex2img.php?eq=W%5E%7B%5Bl%5D%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) matrix corresponding to layer `l` of our network.

## Forward propagation

![img](http://www.sciweavers.org/tex2img.php?eq=Z%5E%7B%5B1%5D%7D%20%3DW%5E%7B%5B1%5D%7DX%2Bb%5E%7B%5B1%5D%7D%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

![img](http://www.sciweavers.org/tex2img.php?eq=A%5E%7B%5B1%5D%7D%3Dg_%7BReLU%28Z%5E%7B%5B1%5D%7D%29%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

![img](http://www.sciweavers.org/tex2img.php?eq=Z%5E%7B%5B2%5D%7D%3DW%5E%7B%5B2%5D%7DA%5E%7B%5B1%5D%7D%2Bb%5E%7B%5B2%5D%7D%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

![img](http://www.sciweavers.org/tex2img.php?eq=A%5E%7B%5B2%5D%7D%3Dg_%7Bsoftmax%28%F0%9D%91%8D%5E%7B%5B2%5D%7D%29%7D%0A%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

### Softmax activation

### Rectified linear unit (ReLU)

## Backward propagation
