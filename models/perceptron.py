import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score,train_test_split 
from sklearn.linear_model import perceptron
from data_preprocessor import get_data

def run_perceptron(x_train, x_test, y_train, y_test):
	clf = perceptron.Perceptron(penalty='l1', tol=None, max_iter=50, verbose=0, random_state=None, fit_intercept=True, eta0=0.02)
	clf.fit(x_train, y_train)
	scores = cross_val_score(clf, x_test, y_test, cv=5)
	print("perceptron: %.15f" % scores.mean())
	with open("output.txt", "a") as text_file:
		print(f"perceptron:",scores.mean(), file=text_file)

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_perceptron(x_train, x_test, y_train, y_test)