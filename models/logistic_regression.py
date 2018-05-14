import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from data_preprocessor import get_data

def run_logistic_regression(x_train, x_test, y_train, y_test):
	clf = linear_model.LogisticRegression(penalty='l1', verbose=0, random_state=None, fit_intercept=True)
	clf.fit(x_train, y_train)
	scores = cross_val_score(clf, x_test, y_test, cv=5)
	print("logistic_regression: %.15f" % scores.mean())
	with open("output.txt", "a") as text_file:
		print(f"logistic_regression:",scores.mean(), file=text_file)

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_logistic_regression(x_train, x_test, y_train, y_test)