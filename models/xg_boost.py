import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from data_preprocessor import get_data

def run_xg_boost(x_train, x_test, y_train, y_test):
	clf = GradientBoostingClassifier()
	clf.fit(x_train, y_train)
	scores = cross_val_score(clf, x_test, y_test, cv=5)
	print("XG boost: %.15f" % scores.mean())

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_xg_boost(x_train, x_test, y_train, y_test)