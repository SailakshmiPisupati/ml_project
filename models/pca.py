import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score,train_test_split 
from sklearn import tree
from sklearn.decomposition import PCA
from data_preprocessor import get_data

# principal component is a linear combination of the original variables

def run_pca(x_train, x_test, y_train, y_test, n_components=2, generate_graph = False):
	pca = PCA(n_components=n_components)
	pca.fit(x_train)
	x_train_new = pca.transform(x_train)
	x_test_new = pca.transform(x_test)
	# print(pca.components_)

	if(generate_graph):
		colors = ['navy', 'turquoise'] # 'darkorange'
		for color, _class in zip(colors, ['male', 'female']):
			# _x = x_train_new[y_train == _class, 0]
			# _y = np.zeros(len(_x))
			# plt.scatter(_x, _y, color=color, alpha=.8, lw=2, label=_class)
		    plt.scatter(x_train_new[y_train == _class, 0], x_train_new[y_train == _class, 1], color=color, alpha=.8, lw=2, label=_class)
		plt.legend(loc='best', shadow=False, scatterpoints=1)
		plt.title('PCA of Voice dataset')
		# plt.show()
		plt.savefig('../outputs/pca.png')

	return x_train_new, x_test_new

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_pca(x_train, x_test, y_train, y_test) # , True



