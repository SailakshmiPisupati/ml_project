import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO

def generate_graph(x_train, y_train, file_name, title, xlabel, ylabel):
	colors = ['navy', 'turquoise'] # 'darkorange'
	for color, _class in zip(colors, ['male', 'female']):
		# _x = x_train[y_train == _class, 0]
		# _y = np.zeros(len(_x))
		# plt.scatter(_x, _y, color=color, alpha=.8, lw=2, label=_class)
	    plt.scatter(x_train[y_train == _class, 0], x_train[y_train == _class, 1], color=color, alpha=.8, lw=2, label=_class)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	# plt.show()
	plt.savefig('../outputs/' + file_name)

if __name__ == '__main__':
	generate_graph()