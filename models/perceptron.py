import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score,train_test_split 
from sklearn.linear_model import perceptron
from data_preprocessor import get_data

def run_perceptron():
	X_train, X_test, y_train, y_test = get_data()
	clf = perceptron.Perceptron(penalty='l1', n_iter=50, verbose=0, random_state=None, fit_intercept=True, eta0=0.02)
	clf.fit(X_train, y_train)
	scores = cross_val_score(clf, X_test, y_test, cv=5)
	print("perceptron: %.15f" % scores.mean())

if __name__ == '__main__':
	run_perceptron()