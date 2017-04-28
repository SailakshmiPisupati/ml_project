import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score,train_test_split 
from sklearn import tree
from data_preprocessor import get_data

def run_decision_tree():
	X_train, X_test, y_train, y_test = get_data()

	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	scores = cross_val_score(clf, X_test, y_test, cv=5)
	print("decision_tree: %.15f" % scores.mean())

	dotfile = StringIO()
	tree.export_graphviz(clf, out_file = dotfile, class_names = ['-', '+'], rounded = True)  #, feature_names = ['1', '0']
	graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
	graph.write_pdf("../outputs/decision_tree.pdf")


if __name__ == '__main__':
	run_decision_tree()