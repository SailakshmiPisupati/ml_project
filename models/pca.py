import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score,train_test_split 
from sklearn import tree
from sklearn.decomposition import PCA
from data_preprocessor import get_data
from graph_generator import generate_graph

# principal component is a linear combination of the original variables

def run_pca(x_train, x_test, y_train, y_test, n_components=2, to_generate_graph = False):
	pca = PCA(n_components=n_components)
	pca.fit(x_train)
	x_train_new = pca.transform(x_train)
	x_test_new = pca.transform(x_test)

	if(to_generate_graph):
		generate_graph(x_train_new, y_train, 'pca.png')

	return x_train_new, x_test_new

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_pca(x_train, x_test, y_train, y_test, 2, True) #



