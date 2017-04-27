import csv
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from StringIO import StringIO
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn import linear_model

with open('voice.csv') as voicefile:
	myFileReader = csv.reader(voicefile, delimiter=',', quotechar='|')
	X = []
	Y = []
	for row in myFileReader:
		X.append(row)

	# preparing attribute and label data
	for i in range(len(X)):
		Y.append(X[i][20])
		del(X[i][20])
	# removing column names
	del(X[0]) 
	del(Y[0])

	X = preprocessing.scale(X) # attribute scaling so that attributes with high values do not dominate
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
	clf = linear_model.LogisticRegression(penalty='l1', verbose=0, random_state=None, fit_intercept=True)
	clf.fit(X_train, y_train)
	scores = cross_val_score(clf, X_test, y_test, cv=5)
	print(scores.mean())
