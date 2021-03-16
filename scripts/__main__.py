import numpy as np
from scripts import NN

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    #This line is code is made for preprocessing and training my neural network implementation
    headers = ['age', 'sex', 'chest_pain', 'resting_blood_pressure',
               'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
               'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', "slope of the peak",
               'num_of_major_vessels', 'thal', 'heart_disease']

    heart_df = pd.read_csv('~/Downloads/heart.dat', sep=' ', names=headers)
    #print(heart_df.head())
    X = heart_df.drop(columns=['heart_disease'])
    heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
    heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)

    y_label = heart_df['heart_disease'].values.reshape(X.shape[0], 1)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2,)

    # standardize the dataset
    sc = StandardScaler()
    sc.fit(Xtrain)
    Xtrain = sc.transform(Xtrain)
    Xtest = sc.transform(Xtest)
    # print(Xtrain[0:10])
    # print(ytrain[0:10])
    # print(len(Xtrain))
    # print(len(ytrain))
    Test=NN.NeuralNetwork(X=Xtrain,Y=ytrain, layers=[8, 3, 8, 1], epoch=100,learningrate=0.1,seed=1)

    # for i in Test.bias:
    #     print(i.shape)
    # print("----------------")
    # for i in Test.weights:
    #     print(i.shape)
    # print("----------------")
    # for i in Test.attributes:
    #     print(i.shape)
    Test.runmodel()


if __name__ == "__main__":
    main()