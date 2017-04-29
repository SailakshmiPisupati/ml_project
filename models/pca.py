import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score,train_test_split 
from sklearn import tree
from sklearn.decomposition import PCA
from data_preprocessor import get_data

def run_pca(x_train, x_test, y_train, y_test):
	pca = PCA(n_components=2)
	pca.fit(x_train)
	x_train_new = pca.transform(x_train) # Apply dimensionality reduction to X
	# print(pca.components_)

	colors = ['navy', 'turquoise'] # 'darkorange'

	for color, _class in zip(colors, ['male', 'female']): 
	    plt.scatter(x_train_new[y_train == _class, 0], x_train_new[y_train == _class, 1], color=color, alpha=.8, lw=2, label=_class)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('PCA of Voice dataset')
	# plt.show()
	plt.savefig('../outputs/pca.png')

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_pca(x_train, x_test, y_train, y_test)



