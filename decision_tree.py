import csv
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from StringIO import StringIO
from sklearn import tree
from sklearn.model_selection import cross_val_score

with open('voice.csv') as voicefile:
	myFileReader = csv.reader(voicefile, delimiter=',', quotechar='|')
	X = []
	Y = []
	for row in myFileReader:
		X.append(row)

	for i in range(len(X)):
		Y.append(X[i][20])
		del(X[i][20])

	print[X[0][5]]
	del(X[0])
	del(Y[0])
	testSetX = []
	testSetY = []
	trainingSetX = []
	trainingSetY = []
	testSize = 0.2*len(X)
	count = 0
	while (count < testSize):
		testSetX.append(X[count])
		testSetY.append(Y[count])
		count = count + 1

	while (count < len(X)):
		trainingSetX.append(X[count])
		trainingSetY.append(Y[count])
		count = count + 1

	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(trainingSetX, trainingSetY)
	scores = cross_val_score(clf, trainingSetX, trainingSetY, cv=5)
	print(scores.mean())
	dotfile = StringIO()
	tree.export_graphviz(clf, out_file = dotfile, class_names = ['-', '+'], rounded = True)  #, feature_names = ['1', '0']
	graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
	graph.write_pdf("clf1.pdf")