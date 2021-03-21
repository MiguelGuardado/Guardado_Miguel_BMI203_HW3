# Project 3 - Neural Networks
## Submission 03/21/2021

![BuildStatus](https://github.com/MiguelGuardado/Guardado_Miguel_BMI203_HW3/workflows/HW3/badge.svg?event=push)

This repo is an implementation of a neural network by hand, only utilizing numpy arrays to train and learn the weights. This Neural Network was made to run neural network 
made for two specific tasks, running autoencoder's, or running classification based off an sigmoid activation function. This implementation is made as a class called NeuralNetwork
which was designed to handle a number of different parameters to train and learn the data. This network weight/bias passing though forward and back propogation was made so you can 
define any amount of layers and not be hardcoded for specific architectures so its scalable for any test desired. Additionally parameters were also inputted, see below and in the docs. This was done as a final project for 
UCSF's Biological and Medical Information Algortihims class for the winter 2021 quarter. 

### docs
This section hold the API documentation for the cluster module and contains all the objects and functions calls in scripts. Please see open the html file found  below to access API docs.

```
open docs/build/html/index.html
```

### Running a Simple Model

To run a 'simple' (its okay if this isn't simple to you :) ) neural network, some key parameters must be specified. You must import the NN.py class into your directory, or know where
it can be found to initialize.  I will run a neural network where I will have 3 hidden layers of size 50,10,5 nodes, and 
I will have 1 output layer that will hold the value of the binary prediction of the observation. I will also train the 
model for 10000 iterations, or epochs, with a learning rate of 0.1 and a random seed of 1. Finally I will use stochastic gradient
descent as my backprop function. Assuming you define X to be all the observation you want to train, and Y is the true prediction 
for each neural network. For my implementation, you dont need to specify the length of the input layer, it will get read in automatically 
by the training data to evaluate

```
from scripts import NN
from scripts import io

Test=NN.NeuralNetwork(X=X,Y=Y, layers=[50,10,5,1], epoch=10000,learningrate=0.1,seed=123,gradient_descent='stochastic' )
```

