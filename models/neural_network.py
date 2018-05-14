import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from data_preprocessor import get_data

def run_neural_network(x_train, x_test, y_train, y_test):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
	clf.fit(x_train, y_train)
	scores = cross_val_score(clf, x_test, y_test, cv=5)
	print("neural_network: %.15f" % scores.mean())
	with open("output.txt", "a") as text_file:
		print(f"neural_network:",scores.mean(), file=text_file)

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_neural_network(x_train, x_test, y_train, y_test)