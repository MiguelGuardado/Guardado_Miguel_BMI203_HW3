import numpy as np
from numpy.random import MT19937
from scripts import io

class NeuralNetwork:
    """

        Attributes:
            X (array-like): Input data you want to train, you can have as many observation as your want, but you must make sure each observation has the same lentgh for the input layer
            Y (array-like): Input data's actuall predictor to use as testing for the network, it can be a single predictions or multuple layer prediction.
            learningrate (int): learning rate of a network, will impact how fast you will try and reach convergence.
            epoch (int): How many iterations you will go though to train the data.
            layers (array-like): array of number of hidden to define for your model, will read input layer from X, will also specify the output layer of the netowrk by the last value in the array.
            gradient_descent(str): type of gradient desncent you want to run. three options are ["stochastic","batch","mini_batch"], Default is
            batch_size (int): If you specify mini_batch as your input parameter, you must include a size of mini batches to run for loss.
            activation_layers (array like): array of activation functions to use for each neural network layer
            weights (array-like): weights of the network to use for training/predicting
            bias (array-like): bias of the network to use for training/predicting
            z (array-like): pre activation values for the hidden and output layer nodes, used as an intermediary for the forward/back prop
            activation (array-like): activation values for the hidden and output layer nodes. The first index of this activation node will be the current input layer we are looking at
            loss_per_epoch (array-like): array of loss of the network for each epoch, array will always be length of epoch.

    """
    def __init__(self, X, Y, layers, epoch, learningrate, seed, gradient_descent="stochastic", batch_size=None, activation_layers=None):
        # set random seed to whatever the user inputs from the Neural Network Initalization.
        np.random.seed(seed)
        self.X=X
        self.Y=Y
        self.learningrate=learningrate
        self.epoch=epoch
        self.layers=layers
        self.gradient_descent=gradient_descent
        self.batch_size=batch_size
        self.loss_per_epoch = None
        #Initialize weights/bias with non zero values of layer dimension weights
        self.loadweights(layers)
        #Initalize empty z and activation layers
        self.loadattributes(layers)



    def loadweights(self, layers):
        """
        Internal functions
        Will load and inistalize the weight and bias for the neural networks to learn. Each weight and bias will be initalized
        randomly to avoid fitting a linear model.

        """
        weights = []
        bias = []
        #This will create the initalization layer based off the size of the input node, and the first hidden layer
        weights.append(np.random.normal(size=(len(self.X[0]), layers[0])))
        bias.append(np.random.normal(size=layers[0]))
        for i in range(1, len(layers)):
            bias.append(np.random.normal(size=layers[i]))
            weights.append(np.random.normal(size = (layers[i-1], layers[i])))
        #Update self parameter
        self.weights=weights
        self.bias=bias



    def loadattributes(self, layers):
        """
        Internal functions:
        This will load the z,a for neural networks. For each layer in forward pass
        """
        activations=[]
        #First layer of the activation will be simply the input observations
        activations.append(self.X)
        z=[]
        for i in range(0, len(layers)):
            tmparray = np.zeros([layers[i]])
            activations.append(tmparray)
            z.append(np.zeros(len(tmparray)))
        z=np.array(z,dtype=object)

        self.activations=activations
        self.z=z


    def MSE(self,y,ypred):
        """
        Internal functions: This will calculate the Mean Squared Error of the prediction value and the actual y values.
        Will still be able to use when you run your network.

            Args:
            y (array-like): 1-2 D array of the same shape from ypred, true prediction of the neural network.
            ypred (array-like): 1-2 D array of same shape from y, predictions from the neural network.
            Returns:
            ymse=(array-like): 1-2 D array of same shape of y,ypred. will return the mean squared error of the two arrays

        """

        return (np.mean((y-ypred)**2))


    def feedforward(self):
        """
        Internal functions:

        """
        #We calculate the first pass by hand in respect to the observation we are looking at, with i denoting the
        for j in range(0, len(self.activations)-1):
            #Will use matrix multiplication to get the neuron activation value of Hidden/Output layer.
            self.z[j]=np.matmul( self.activations[j],self.weights[j]) + self.bias[j]

            #Will input the raw calculation into sigmoid function to get 0,1 prediction for each neuron.
            self.activations[j+1]=io.sigmoid_activation(self.z[j])


    def backprop(self):
        """
        Internal function: Backpropgation of the network to update the weights and biases. Will do this iteratively
        so its not static for any specific network architecture.
        """
        #Calculate the Initial error and delta for the model.
        dl_wrt_yhat = (self.y - self.activations[-1])

        #Different cost function for error
        #dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(self.Y, self.eta(self.activations[-1]))

        dl_wrt_sig = io.sigmoid_activation(self.activations[-1],dx=True)
        dl_wrt_zn = np.dot(dl_wrt_yhat , dl_wrt_sig)

        dl_wrt_A = np.dot(dl_wrt_zn,self.weights[-1].T)
        dl_wrt_wn = self.activations[-2].T.dot(dl_wrt_zn)
        dl_wrt_bn = np.sum(dl_wrt_zn, axis=0, keepdims=True)

        self.weights[-1]=self.weights[-1] + self.learningrate*dl_wrt_wn
        self.bias[-1]= self.bias[-1] + self.learningrate*dl_wrt_bn

        for i in reversed(range(len(self.layers)-1)):
            dl_wrt_z = dl_wrt_A * io.sigmoid_activation(self.z[i],dx=True)
            dl_wrt_w = np.dot(self.activations[i].T,dl_wrt_z)
            dl_wrt_b = np.sum(dl_wrt_z, axis=0, keepdims=True)

            self.weights[i] = self.weights[i] + self.learningrate * dl_wrt_w
            self.bias[i] = self.bias[i] +  self.learningrate * dl_wrt_b
            dl_wrt_A = np.dot(dl_wrt_z , self.weights[i].T)


    def getbatch(self):
        """
            Internal function. This will be used to determine the type of batch the network is using to feed the number
            of observation to train on per network
        """
        if (self.gradient_descent== 'mini_batch'):
            batch = np.random.choice(len(self.X), self.batch_size, replace=False)
            batch_train = self.X[batch]
            batch_test = self.Y[batch]
            return(batch_train,batch_test)
        elif(self.gradient_descent == 'batch'):
            batch_train = self.X
            batch_test = self.Y
            return (batch_train, batch_test)
        else:
            print("Gradient Descent method not found, breaking from model training. ")
            exit(0)



    def runmodel(self):
        """

            This is used to run and train the model based on the inputted test data. This will run based on how many epoch
            you will define earlier. This will test the weights and bias and update based off the error. This will utilize
            the forward propogation function, the backpropgation function, and the MSE function to fit the model.
            You must run this function before you use the predict function to make prediction off different datasets.
            Basic workflow of this function.
            1. Iterate though each epoch
                2. Get input observation to run, [batch, mini batch, or stochastic]
                3. Feedforward based on input data
                4. Update weight and bias based off loss of trained values, run backpropogation
                5. Calculate loss of all observations used in the epoch

        """
        loss_per_epoch=[]

        #Will iterate through all epochs,
        for epoch in range(0,self.epoch):
            #Will run stochastic gradient descent by default, but will also check for user input
            if (self.gradient_descent== 'stochastic'):
                #One epoch iterates and updates the weight for each observation in the dataset.
                for i in range(0,len(self.X)):

                    self.x=self.X[i]
                    self.activations[0] = self.x.reshape(1, (len(self.x)))
                    self.y=self.Y[i]
                    #feed forward
                    self.feedforward()
                    #backprop
                    self.backprop()

            else:
                #Return either a full batch of data to train at one pass, or a small subset defined by the user
                batch_train,batch_test=self.getbatch()
                self.x=batch_train
                self.y=batch_test
                self.activations[0] = self.x
                self.feedforward()
                self.backprop()
            #Calulcate Loss for each function
            cost = self.MSE(self.y, self.activations[-1])
            loss_per_epoch.append(cost)
            #Returns an array of loss per epoch,
            self.loss_per_epoch=loss_per_epoch




    def predict(self,X):
        """
        Returns an single array, of predictions based on input data given. This functions should not be called
        Until you train the NN model via self.runmodel() method. This data can contain any desired amount of
        observation to predict, but each observation input layer MUST be the same size of the model you
        rained to run.

        Args:
        X (array-like): Input layer for the neural network to calculate predictions on

        Returns:
        Ypred (array-like): Activation function predictions based on the training data.

        """
        self.activations[0]=X
        self.feedforward()
        predictions=self.activations[-1]
        predictions=predictions[0]

        return (predictions)


