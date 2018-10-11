from sklearn.tree import DecisionTreeClassifier
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from read_data import preprocess

print('Default scores:')
trained = preprocess("dtree_data/train.csv")
X = [x[:-1] for x in trained]
Y = [x[-1] for x in trained]
clf = DecisionTreeClassifier(random_state=0)
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

print('Scores when maxdepth=6:')
clf = DecisionTreeClassifier(random_state=0,max_depth=4)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)



print('Scores when maxdepth=6:')
clf = DecisionTreeClassifier(random_state=0,max_depth=6)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when maxdepth=8:')
clf = DecisionTreeClassifier(random_state=0,max_depth=8)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when maxdepth=10:')
clf = DecisionTreeClassifier(random_state=0,max_depth=10)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when maxdepth=12:')
clf = DecisionTreeClassifier(random_state=0,max_depth=12)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)



print('Scores when min sample split=2:')
clf = DecisionTreeClassifier(random_state=0,min_samples_split=2)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when min sample split=3:')
clf = DecisionTreeClassifier(random_state=0,min_samples_split=3)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when min sample split=4:')
clf = DecisionTreeClassifier(random_state=0,min_samples_split=4)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when min sample split=6:')
clf = DecisionTreeClassifier(random_state=0,min_samples_split=6)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when min_samples_leaf=1:')
clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=20)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when min_samples_leaf=10:')
clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=20)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when min_samples_leaf=20:')
clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=20)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when min_samples_leaf=50:')
clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=50)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when min_samples_leaf=100:')
clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=100)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)


print('Scores when min_samples_leaf=200:')
clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=200)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when min_samples_leaf=1000:')
clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=500)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('Scores when min_samples_leaf=2000:')
clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=500)
clf.fit(X,Y)
validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)

print('')
print('Scores ideal:')
clf = DecisionTreeClassifier(random_state=0,min_samples_split=3,max_depth=8)
clf.fit(X,Y)
training_score = clf.score(X,Y)
print('Training set score:',training_score)
testing_score = clf.score(test_X,test_Y)
print('Test set score:',testing_score)

validing_score = clf.score(valid_X,valid_Y)
print('Validation set score:',validing_score)