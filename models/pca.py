import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score,train_test_split 
from sklearn import tree
from data_preprocessor import get_data
from sklearn.decomposition import PCA

def run_pca():
	X_train, X_test, y_train, y_test = get_data()
	# X_train = np.array(X_train)
	# X_test = np.array(X_test)
	y_train = np.array(y_train)
	# y_test = np.array(y_test)

	pca = PCA(n_components=2)
	pca.fit(X_train)
	X_train_new = pca.transform(X_train) # Apply dimensionality reduction to X
	# print(pca.components_)

	colors = ['navy', 'turquoise'] # 'darkorange'

	for color, _class in zip(colors, ['male', 'female']): 
	    plt.scatter(X_train_new[y_train == _class, 0], X_train_new[y_train == _class, 1], color=color, alpha=.8, lw=2, label=_class)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('PCA of Voice dataset')
	# plt.show()
	plt.savefig('../outputs/pca.png')

if __name__ == '__main__':
	run_pca()



