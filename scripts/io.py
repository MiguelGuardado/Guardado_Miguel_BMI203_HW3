import numpy as np
import scipy.special as sci
import random

def relu(x, dx = False):
    '''
        Implementation of rectified linear activation functions for neural network node activation, as well evaluating the derivative of the
        function. relu activation is a piecewise linear function that will output the same input if the value is positive
        ,otherwise, it will output zero. This activation function is advantagous to avoid issues of saturation and does a
        better job training the neural network due to the avoidance of saturation and sensitivity

        Args:
        x (array-like): Values you want to calculate the activation function on, will calculate each value individually
        dx (boolean): if you want to calculate the activation function or its derivative, default is false, but if true
        than it will calculate the derivative of the activation function.

        Returns:
        x_activation (array-like): Array of the same shape as the input, will be the activation values for each inputted node
    '''
    if dx:

        return (x > 0).astype(int)
    else:
        return np.maximum(0, x)


def sigmoid_activation(x, dx = False):
    '''
        Implementation of sigmoid activation functions for neural network node activation, as well evaluating the derivative of the
        function. Sigmoid activation has traditionally has been used of neural network learning, where you predict the
        probability as an output. Since the probability of anything will exist between the range of 0 and 1. this is often
        used for probability predicitng of an activation function.

        Args:
        x (array-like): Values you want to calculate the activation function on, will calculate each value individually
        dx (boolean): if you want to calculate the activation function or its derivative, default is false, but if true
        than it will calculate the derivative of the activation function.

        Returns:
        x_activation (array-like): Array of the same shape as the input, will be the activation values for each inputted node
    '''
    sig = sci.expit(x + 1)
    if dx:
        return sig*(1-sig)
    else:
        return sig

def read_fasta(filename):
    """
        This function is used to read in the fasta sequence text file. This will read in the file name and read and load
        in the file, make sure to check the directory and if it exist. if error is thrown, then you must
        specify the input filename.

        Args:
        filename (str): Fasta Filepath to find the sequence data to read in.

        Returns:
        seq(array-like): Array of the datatype that read from the file, will return each line as is its row in the array.

    """
    with open(filename, "r") as f:
        next(f)

        seq=[]
        single_seq=[]
        for line in f:
            line=line.strip('\n')
            first_index=list(line)
            if first_index[0] == ">":
                single_seq_merged="".join(single_seq)
                seq.append(single_seq_merged)
                single_seq=[]
            else:
                single_seq.append(line)
        seq=np.array(seq)
    return(seq)


def read_seq_txts(filename):
    """
        This function is used to read in the sequence text functions. Different from a fasta file, this is just the raw
        dna information for each individual, with each line representing an different seqeunce.

        Args:
        filename (str): Filepath to find the sequence data to read in.

        Returns:
        seq(array-like): Array of the datatype that read from the file, will return each line as is its row in the array.

    """
    seq=[]
    with open(filename,"r") as f:
        for line in f:
            line = line.strip('\n')
            seq.append(line)
    seq=np.array(seq)
    return (seq)


def get_dataset(pos_seq,neg_seq,neq_keep):
    """
            This function is used to extract a training dataset for the postive and negative controls. The negative
            controls in the transcription binding data is unbalanaced and need us to downsample the number of training
            points used. This will input the two pos and neg arrays, as well as a number of how many negative samples
            you want to keep.

            Args:
            pos_seq (array-like): Input layer for the neural network to calculate predictions on
            neg_seq (array-like):
            neg_keep (int): number of negative controls to keep for the imbalanced dataset.

            Returns:
            X (array-like): Training X set to run
            Y (array-like): Training Y set to run

    """
    X=[]
    Y=[]
    for i in pos_seq:
        pos_one_hot=dna_one_hot(i)
        X.append(pos_one_hot.onehot)
        Y.append(1)
    #This will randomize the order of negative controls controls to use
    random.shuffle(neg_seq)

    # This will intialize the postive controls, or regions that are trancription factor binding sites.
    neg_site_count = 0
    for i in neg_seq:
        if ( neg_site_count >= neq_keep):
            break
        raw_seq_sample=list(i)
        #For each sequence, I will subset a random number which reprsents the index where the sample will start
        sample_start_idx=random.randint(0,len(raw_seq_sample)-17)
        neg_seq_sub="".join(raw_seq_sample[sample_start_idx:sample_start_idx+17])
        #print(neg_seq_sub not in pos_seq)
        if (neg_seq_sub not in pos_seq):
            neg_site_count += 1
            neg_seq_onhot = dna_one_hot(neg_seq_sub)
            X.append(neg_seq_onhot.onehot)
            Y.append(0)
    X=np.array(X)
    Y=np.array(Y)
    return(X,Y)






class dna_one_hot:
    """
    This class is used to hold and calculate the one hot encodings for a dna sequences. All that is needed for
    initialization is the dna sequence you are trying to encode. This is used to input the dna sequence into the
    neural network.

    Attributes:
    raw_sequence (array-like): Hold the raw seqeunce of the file being inputted for one hot encoding
    sequence_list (array-like): list of each base pair in the index, in case you want to subset sections of the sequence for input.
    onehot (array-like): Binary array of DNA sequence encoding. array will be a of size 4 * length of the sequence.
    """
    def __init__(self,fasfa):
        self.raw_sequence= fasfa
        self.sequence_list = list(fasfa)

        onehot=[]
        for i in list(fasfa):
            if i == 'A':
                onehot.append([1,0,0,0])
            if i == 'T':
                onehot.append([0,1,0,0])
            if i == 'G' :
                onehot.append([0,0,1,0])
            if i == 'C':
                onehot.append([0,0,0,1])

        #Flatten an array into a single list.
        onehot = np.array([item for sublist in onehot for item in sublist])
        self.onehot= onehot



