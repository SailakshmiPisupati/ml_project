import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from data_preprocessor import get_data

def run_logistic_regression():
	X_train, X_test, y_train, y_test = get_data()
	clf = linear_model.LogisticRegression(penalty='l1', verbose=0, random_state=None, fit_intercept=True)
	clf.fit(X_train, y_train)
	scores = cross_val_score(clf, X_test, y_test, cv=5)
	print("logistic_regression: %.15f" % scores.mean())

if __name__ == '__main__':
	run_logistic_regression()