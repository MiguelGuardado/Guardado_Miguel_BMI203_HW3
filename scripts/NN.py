import numpy as np
import math
import scipy.special as sci
from random import sample
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

class NeuralNetwork:
    # def __init__(self, setup=[[68,25,"sigmoid",0],[25,1,"sigmoid",0]],lr=.05,seed=1,error_rate=0,bias=1,iter=500,lamba=.00001,simple=0):
    #     #Note - these paramaters are examples, not the required init function parameters
    #     pass
    def __init__(self, X, Y, layers, epoch,learningrate,seed):
        # will set the python seed to be.
        np.random.seed(seed)
        self.raw_data=X
        self.raw_outputs=Y
        self.x=X
        self.Y=Y
        self.learningrate=learningrate

        #I will initialize weights for a node
        #print(weights)
        self.epoch=epoch
        self.loss=[]
        self.layers=layers
        self.loadweights(layers)
        self.loadattributes(layers)


    """
    this will be used to load the weight matrices for each layer, I design my weight matrices such that is a 3d matrix
    where the first dimension of the matrix is just a list that holds the each input layer and weights by index, 
    the hidden layers/weights, up to the prediction of the layer. The second and third dimension is a 2D 
    np matrix of the weights of the current feature. 
    ex: InputData has 14 observations, setup=[8,3,8,1]
    Return [[8x14],[3X8],[8X3],[1X18]]
    
    """
    def loadweights(self, layers):
        weights = []
        weights.append(np.random.normal(size=(len(self.x[0]),layers[0]),scale=0.01))
        for i in range(1, len(layers)):
            weights.append(np.random.normal(size = (layers[i-1], layers[i]),scale=0.01))
        # for weight in weights:
        #     print(weight)
        self.weights=weights
    """
    Attributes will hold the values of the activation function output where we determine wether a neuron is going to 
    be activated or not based on the previous layers' weight and attributes. The size of the array being returned is
    len(setup) + 2 for which we will have a neuron layer for the input/output layer + the hidden layers that we defined.
    """
    def loadattributes(self, layers):
        activations=[]
        bias=[]
        z=[]
        for i in range(0, len(layers)):
            tmparray = np.zeros([layers[i]])
            activations.append(tmparray)
            bias.append(np.random.normal(size=len(tmparray),scale=0.01))
            z.append(np.zeros(len(tmparray)))
        z=np.array(z,dtype=object)
        self.activations=activations
        self.bias=bias
        self.z=z


    def MSE(self,y,ypred):
        N=len(y)
        return (np.sum((y-ypred)**2)/N)


    def feedforward(self):
        # delta=[]
        # bias=[]
        # for i in self.layers:
        #     delta.append(np.zeros(i))
        # for i in self.bias:
        #     bias.append(np.zeros(i.shape))


        observation=self.x
        #We calculate the first pass by hand in respect to the observation we are looking at, with i denoting the
        # iterations of observation we are observing.
        self.z[0]=np.matmul(observation,self.weights[0])
        self.activations[0] = sigmoid_activation(self.z[0])




        for j in range(1, len(self.activations)):
            #Will use matrix multiplication to get the neuron activation value of Hidden/Output layer.
            # print("-------------------------")
            # print("a",self.activations[j-1].shape)
            # print("w",self.weights[j].shape)
            # print("b",self.bias[j].shape)

            self.z[j]=np.matmul( self.activations[j-1],self.weights[j]) + self.bias[j]



            #append the raw value before sigmoid function to z for backpropcaluations
            #Will input the raw calculation into sigmoid function to get 0,1 prediction for each neuron.
            self.activations[j]=sigmoid_activation(self.z[j])

            #Append the predicted value to a ypred list to calculate the total loss for each node.
        yhat=self.activations[-1]
        #Now we will calculate the delta loss for each for each individual node for this batch were running
        #single_deltas,single_bias=self.backprop(observation,self.Y[i],ypred[i],self.weights,self.z)

            # for i in range(0,len(delta)):
            #     delta[i]+=single_deltas[i]
            #     bias[i]+=single_bias[i]


        # loss = self.MSE(self.y,ypred)

        return yhat

    """
    This will update the front and back propigation based on mini-batches.
    """
    def backprop(self,yhat):
        weight_delta=[]
        bias_delta=[]

        for i in range(1,len(self.z)):
            self.z[i]=np.reshape(self.z[i],newshape=(1,len(self.z[i])))

        #For the output layer, in our case a single node based on a sigmoid/binary binary prediction. Can be flexible for more output layers
        error=self.y-yhat
        delta_out=error*sigmoid_activation(yhat,dx=True)
        self.weights[3]+=np.dot(self.activations[3].T,delta_out) * self.learningrate
        self.bias[3]+=delta_out*self.learningrate

        slope_hidden2=sigmoid_activation(self.activations[2],dx=True)
        delta_hidden2=np.dot(delta_out,self.weights[3].T)  * slope_hidden2
        self.weights[2]+=np.dot(self.activations[2].T,delta_hidden2) * self.learningrate
        self.bias[2]+=delta_hidden2 * self.learningrate

        slope_hidden3 = sigmoid_activation(self.activations[1], dx=True)

        delta_hidden3 = np.dot(delta_hidden2, self.weights[2].T) * slope_hidden3
        self.weights[1] += np.dot(self.activations[1].T, delta_hidden3) * self.learningrate
        self.bias[1] += delta_hidden3 * self.learningrate

        slope_hidden4 = sigmoid_activation(self.activations[0], dx=True)
        delta_hidden4 = np.dot(delta_hidden3, self.weights[1].T) * slope_hidden4
        self.weights[0] += np.dot(self.activations[0].T, delta_hidden4) * self.learningrate
        self.bias[0] += delta_hidden4 * self.learningrate


    def runmodel(self):
        loss_per_epoch=[]

        for epoch in range(0,self.epoch):
            Ypred=[]
            for i in range(0,len(self.raw_outputs)):
                self.x=self.raw_data[i]
                self.y=self.raw_outputs[i]

                yhat=self.feedforward()
                Ypred.append(yhat)

                self.backprop(yhat)
            # for i in range(0,len(self.weights)):
            #     self.weights[i]=self.weights[i] - (1/N) * delta[i]
            #     self.bias[i]=self.bias[i] - (1/N) * bias[i]


            print(self.MSE(self.Y,Ypred))
            #self.backprop(ypred)
            # loss_per_epoch.append(loss)
            # print("Loss per Epoch: "+ str(loss))



    def predict(self):
        pass




def sigmoid_activation(x, dx = False):
    '''
    Sigmiod based activation function - as discussed by Mike Keiser in class
    Input: value to be 'activated' can be int, float or array, boolean of if we want the derivative or not
    Output: sigmoid activation of input, derivative if requested
    using expit from scipy to prevent runtime overflow errors
    '''
    sig = sci.expit(x + 1)
    if dx:
        return sig*(1-sig)
    else:
        return sig

def relu(z, dx = False):
    '''
    The ReLu activation function is to performs a threshold
    operation to each input element where values less
    than zero are set to zero.
    '''
    if dx:

        return (z > 0).astype(int)
    else:
        return np.maximum(0, z)