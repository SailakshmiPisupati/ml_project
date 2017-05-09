import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score,train_test_split 
from sklearn import tree
from data_preprocessor import get_data, get_labels
from graph_generator import generate_graph

def run_decision_tree(x_train, x_test, y_train, y_test, generate_graph = False, max_depth = 10):
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(x_train, y_train)
	scores = cross_val_score(clf, x_test, y_test, cv=5)
	print("decision_tree: %.15f" % scores.mean())

	if(generate_graph):
		dotfile = StringIO()
		tree.export_graphviz(clf, out_file = dotfile, class_names = ['female', 'male'], rounded = True, feature_names = get_labels(), max_depth = max_depth)
		graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
		graph.write_pdf("../outputs/decision_tree.pdf")

def plot_top_2_features(x_train, x_test, y_train, y_test):
	generate_graph(x_train, y_train, 'dt_top_2.png', 'Plot of Top 2 features of Voice Dataset')


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	# run_decision_tree(x_train, x_test, y_train, y_test, True, 5)
	plot_top_2_features(x_train, x_test, y_train, y_test)