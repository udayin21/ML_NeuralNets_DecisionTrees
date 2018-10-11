from sklearn.ensemble import RandomForestClassifier
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from read_data import preprocess

print('Default scores:')
trained = preprocess("dtree_data/train.csv")
X = [x[:-1] for x in trained]
Y = [x[-1] for x in trained]
clf = RandomForestClassifier(random_state=0)
clf.fit(X,Y)
training_score = clf.score(X,Y)
print('Training set score:',training_score)

tested = preprocess("dtree_data/test.csv")
test_X = [x[:-1] for x in tested]
test_Y = [x[-1] for x in tested]
testing_score = clf.score(test_X,test_Y)
print('Testing set score:',testing_score)

valided = preprocess("dtree_data/valid.csv")
valid_X = [x[:-1] for x in valided]
valid_Y = [x[-1] for x in valided]
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when nestimators=5:')
clf = RandomForestClassifier(random_state=0,n_estimators=5)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when nestimators=1:')
clf = RandomForestClassifier(random_state=0,n_estimators=1)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when nestimators=2:')
clf = RandomForestClassifier(random_state=0,n_estimators=2)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when nestimators=8:')
clf = RandomForestClassifier(random_state=0,n_estimators=8)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when nestimators=10:')
clf = RandomForestClassifier(random_state=0,n_estimators=10)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when nestimators=15:')
clf = RandomForestClassifier(random_state=0,n_estimators=15)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when nestimators=30:')
clf = RandomForestClassifier(random_state=0,n_estimators=20)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when max featrues=1:')
clf = RandomForestClassifier(random_state=0,max_features=1)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when max featrues=2:')
clf = RandomForestClassifier(random_state=0,max_features=2)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when max featrues=4:')
clf = RandomForestClassifier(random_state=0,max_features=4)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when max featrues=6:')
clf = RandomForestClassifier(random_state=0,max_features=6)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when max featrues=8:')
clf = RandomForestClassifier(random_state=0,max_features=8)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when max featrues=10:')
clf = RandomForestClassifier(random_state=0,max_features=10)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when bootstrap=True:')
clf = RandomForestClassifier(random_state=0,bootstrap=True)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when bootstrap=False:')
clf = RandomForestClassifier(random_state=0,bootstrap=False)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print(' ')
print('Scores ideal:')
clf = RandomForestClassifier(random_state=0,bootstrap=True,n_estimators=15)
clf.fit(X,Y)
training_score = clf.score(X,Y)
print('Training set score:',training_score)
testing_score = clf.score(test_X,test_Y)
print('Testing set score:',testing_score)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)