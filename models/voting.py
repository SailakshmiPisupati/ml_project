
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from io import StringIO
from sklearn.model_selection import cross_val_score
from data_preprocessor import get_data
from sklearn.model_selection import cross_val_score,train_test_split 
from sklearn import tree
from sklearn import neighbors
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

def run_voting(x_train, x_test, y_train, y_test):
	models = getModels(x_train, x_test, y_train, y_test)
	correct_classification_count = 0
	for i in range(len(x_test)):
		count = 0;
		for j in range(len(models)):
			instance = x_test[i].reshape(1,-1)
			if(models[j].predict(instance) == y_test[i]):
				count = count + 1
			else:
				count = count - 1
		if(count > 0):
			correct_classification_count  =  correct_classification_count + 1
	accuracy = (correct_classification_count*1.0)/len(x_test)
	print("Voting: %.15f" % accuracy)
	with open("output.txt", "a") as text_file:
		print(f"Voting:",accuracy, file=text_file)

def getModels(x_train, x_test, y_train, y_test):
	clf1 = tree.DecisionTreeClassifier()
	clf1 = clf1.fit(x_train, y_train)

	clf2 = neighbors.KNeighborsClassifier(10, 'uniform')
	clf2 = clf2.fit(x_train, y_train)

	clf3 = linear_model.LogisticRegression(penalty='l1', verbose=0, random_state=None, fit_intercept=True)
	clf3 = clf3.fit(x_train, y_train)

	clf4 = GaussianNB()
	clf4 = clf4.fit(x_train, y_train)
	
	clf5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
	clf5 = clf5.fit(x_train, y_train)

	clf6 = perceptron.Perceptron(penalty='l1', tol=None, max_iter=50, verbose=0, random_state=None, fit_intercept=True, eta0=0.02)
	clf6 = clf6.fit(x_train, y_train)

	clf7 = RandomForestClassifier(n_estimators=10)
	clf7 = clf7.fit(x_train, y_train)

	clf8 = svm.SVC()
	clf8 = clf8.fit(x_train, y_train)

	clf9 = GradientBoostingClassifier()
	clf9 = clf9.fit(x_train, y_train)

	return [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9]


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	run_voting(x_train, x_test, y_train, y_test)
