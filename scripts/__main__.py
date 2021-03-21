from scripts import NN
from scripts import io

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

def main():
    np.random.seed(2)
    #Part 1, Autoencoder Implementation
    Xtrain=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                    [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])

    Ytrain = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                      [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])

    Test=NN.NeuralNetwork(X=Xtrain,Y=Ytrain, layers=[3,8], epoch=1000,learningrate=0.1,seed=1)
    Test.runmodel()
    plt.plot(Test.loss_per_epoch)
    plt.title("Training of AutoEncoder on 8x8  Identity Matrix")
    plt.ylabel('Loss(MSE)')
    plt.show()
    exit()


    #Part 2a, first I will load the positive and negative data for training/testing
    neg_seq=io.read_fasta("../data/yeast-upstream-1k-negative.fa")
    pos_seq=io.read_seq_txts("../data/rap1-lieb-positives.txt")

    X,Y=io.get_dataset(pos_seq,neg_seq,500)

    X_train, X_test, y_train, y_test = train_test_split(X,Y)

    Test=NN.NeuralNetwork(X=X_train,Y=y_train, layers=[17,4,1], epoch=1000, learningrate=0.01 , seed=80, batch_size=5)
    Test.runmodel()
    prediction=Test.predict(X_test)

    #Part 4a Provide an example of the input and output for one true positive sequence and one true negative sequence
    for i in range(0,len(X_test)):
        print("Input Seq",X_test[i],"actual",y_test[i],"pred",prediction[i])

    #Part 4c Stop criterion for convergence in my learned parameters
    Test=NN.NeuralNetwork(X=X_train,Y=y_train, layers=[17,4,1], epoch=1000, learningrate=0.01 , seed=80, batch_size=5)
    Test.runmodel()
    prediction=Test.predict(X_test)
    fpr,tpr,threshold = roc_curve(y_test,prediction)
    roc_auc=auc(fpr,tpr)
    print(roc_auc)
    plt.plot(Test.loss_per_epoch)
    plt.title("Training for regions of trancription factor binding sites (500 Negative Contols)")
    plt.ylabel('Loss(MSE)')
    plt.show()

    #Part 5, k-fold cross validation
    X, Y = io.get_dataset(pos_seq, neg_seq, 503)
    X,Y=io.get_dataset(pos_seq,neg_seq,500)
    X_train, X_test, y_train, y_test = train_test_split(X,Y)
    K=[2,5,10,15,20]
    for k in K:
        print("Running k fold test with a k=",k)
        cv = KFold(n_splits=k,random_state=1,shuffle=True)
        auroc_accuracy=[]
        for train, test in cv.split(X):
            # print('train: %s, test: %s' % (X[train].shape, X[test].shape))
            # print('train: %s, test: %s' % (Y[train].shape, Y[test].shape))
            kfold=NN.NeuralNetwork(X=X[train],Y=Y[train], layers=[17,4,1], epoch=200, learningrate=0.01 , seed=420, batch_size=5)
            kfold.runmodel()
            ypred=kfold.predict(X[test])
            fpr,tpr,threshold = roc_curve(Y[test],ypred)
            roc_auc=auc(fpr,tpr)
            auroc_accuracy.append(roc_auc)
            del kfold,roc_auc,tpr,fpr,ypred
        print("Average AuRoc for k fold trial: ",np.mean(auroc_accuracy))


    X, Y = io.get_dataset(pos_seq, neg_seq, 503)
    k_roc=[]

    for i in range(0,200):
        print(i)
        cv = KFold(n_splits=5, random_state=i, shuffle=True)
        auroc_accuracy = []
        for train, test in cv.split(X):
            kfold = NN.NeuralNetwork(X=X[train], Y=Y[train], layers=[17, 4, 1], epoch=200, learningrate=0.01, seed=i, batch_size=5)
            kfold.runmodel()
            ypred = kfold.predict(X[test])
            fpr, tpr, threshold = roc_curve(Y[test], ypred)
            roc_auc = auc(fpr, tpr)
            auroc_accuracy.append(auroc_accuracy)
        k_roc.append(np.mean(roc_auc))

    print("Mean k=5 fold auroc after 200 iterations",np.mean(k_roc))

    #Part 4 Grid Search Implementation
    X, Y = io.get_dataset(pos_seq, neg_seq, 503)
    learning_rate=[0.01,0.1,0.2,0.5,1]
    layers=[[50,17,1],[10,4,1],[17,4,1],[10,1],[4,1]]
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    cnt_i=0
    grid_search=np.zeros((5,5))
    for i in learning_rate:
        cnt_j=0
        for j in layers:
            print(cnt_i)
            print(cnt_j)
            kfold = NN.NeuralNetwork(X=X_train, Y=y_train, layers=j, epoch=200, learningrate=i, seed=421, batch_size=5)
            kfold.runmodel()
            ypred = kfold.predict(X_test)
            fpr, tpr, threshold = roc_curve(y_test, ypred)
            roc_auc = auc(fpr, tpr)
            grid_search[cnt_i,cnt_j]=roc_auc
            print(grid_search)
            cnt_j = cnt_j + 1
        cnt_i= cnt_i+1

    #Testing the updated function
    X,Y=io.get_dataset(pos_seq,neg_seq,500)

    X_train, X_test, y_train, y_test = train_test_split(X,Y)

    Test=NN.NeuralNetwork(X=X_train,Y=y_train, layers=[17,4,1],
                          gradient_desent='stochastic', epoch=10000, learningrate=0.01 , seed=10)
    Test.runmodel()

    plt.plot(Test.loss_per_epoch)
    plt.title("Training via stochastic gradient decsent (500 Negative Contols)")
    plt.ylabel('Loss (MSE)')
    plt.show()

    #Test two, training the model with fillbatch gradient descent
    X,Y=io.get_dataset(pos_seq,neg_seq,500)

    X_train, X_test, y_train, y_test = train_test_split(X,Y)


    Test=NN.NeuralNetwork(X=X_train,Y=y_train, layers=[17,4,1],
                          gradient_desent='batch', epoch=10000, learningrate=0.01 , seed=10)
    Test.runmodel()

    plt.plot(Test.loss_per_epoch)
    plt.title("Training via full batch gradient decsent (500 Negative Contols)")
    plt.ylabel('Loss (MSE)')
    plt.show()
    #
    #
    # #
    X,Y=io.get_dataset(pos_seq,neg_seq,500)

    X_train, X_test, y_train, y_test = train_test_split(X,Y)

    Test=NN.NeuralNetwork(X=X_train,Y=y_train, layers=[17,4,1],
                          gradient_desent='mini_batch',batch_size=2, epoch=40000, learningrate=0.01 , seed=10)
    Test.runmodel()

    plt.plot(Test.loss_per_epoch)
    plt.title("Training via mini batch (10) gradient decsent (500 Negative Contols)")
    plt.ylabel('Loss (MSE)')
    plt.show()


    #Part 5 Predictions, Show Time!!!
    X,Y=io.get_dataset(pos_seq,neg_seq,500)

    FinalModel=NN.NeuralNetwork(X=X,Y=Y, layers=[17,4,1],
                          gradient_desent='stochastic', epoch=100, learningrate=0.01 , seed=12)
    FinalModel.runmodel()

    rap1_seq = io.read_seq_txts("../data/rap1-lieb-test.txt")

    rap1_seq_predictions=[]
    for seq in rap1_seq:
        seq_encoded=io.dna_one_hot(seq)
        # print(seq_encoded.onehot)
        prediction=FinalModel.predict(seq_encoded.onehot)
        prediction= np.where(prediction > 0.5, 1, 0)
        row=[seq_encoded.raw_sequence,prediction[0]]
        rap1_seq_predictions.append(row)

        del seq_encoded

    rap1_seq_predictions = np.array(rap1_seq_predictions)

    np.savetxt("../data/rap1_seq_predictions.txt",rap1_seq_predictions,fmt='%s')
    #Thanks yall!

if __name__ == "__main__":
    main()