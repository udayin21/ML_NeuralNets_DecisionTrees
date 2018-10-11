from sklearn import linear_model
import csv
import matplotlib.pyplot as plt
import numpy as np
from visualization import *
from numpy import genfromtxt

X = genfromtxt('toy_data/toy_trainX.csv', delimiter=',')       
Y = genfromtxt('toy_data/toy_trainY.csv', delimiter=',')
logreg = linear_model.LogisticRegression(C=1)
logreg.fit(X, Y)
train_score = logreg.score(X,Y)
test_X = genfromtxt('toy_data/toy_testX.csv', delimiter=',')       
test_Y = genfromtxt('toy_data/toy_testY.csv', delimiter=',')   
test_score = logreg.score(test_X,test_Y)
print('Train score:',train_score)
print('Test score',test_score)

plot_decision_boundary(logreg.predict,test_X,test_Y)
