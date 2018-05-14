import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from data_preprocessor import get_data

def run_random_forest(x_train, x_test, y_train, y_test):
	clf = RandomForestClassifier(n_estimators=10)
	clf.fit(x_train, y_train)
	scores = cross_val_score(clf, x_test, y_test, cv=5)
	print("random_forest: %.15f" % scores.mean())
	with open("output.txt", "a") as text_file:
		print(f"random_forest:",scores.mean(), file=text_file)

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_random_forest(x_train, x_test, y_train, y_test)