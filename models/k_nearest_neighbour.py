import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from data_preprocessor import get_data

def run_k_nearest_neighbour(x_train, x_test, y_train, y_test):
	clf = neighbors.KNeighborsClassifier(10, 'uniform')
	clf.fit(x_train, y_train)
	scores = cross_val_score(clf, x_test, y_test, cv=5)
	print("k_nearest_neighbour: %.15f" % scores.mean())
	with open("output.txt", "a") as text_file:
		print(f"k_nearest_neighbour:",scores.mean(), file=text_file)

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_k_nearest_neighbour(x_train, x_test, y_train, y_test)