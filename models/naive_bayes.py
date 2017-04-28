import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from data_preprocessor import get_data

def run_naive_bayes():
	X_train, X_test, y_train, y_test = get_data()

	clf = GaussianNB()
	clf.fit(X_train, y_train)
	scores = cross_val_score(clf, X_test, y_test, cv=5)
	print("naive_bayes: %.15f" % scores.mean())

if __name__ == '__main__':
	run_naive_bayes()