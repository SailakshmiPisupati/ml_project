from decision_tree import run_decision_tree 
from k_nearest_neighbour import run_k_nearest_neighbour
from logistic_regression import run_logistic_regression
from naive_bayes import run_naive_bayes
from neural_network import run_neural_network
from perceptron import run_perceptron
from random_forest import run_random_forest
from svm import run_svm
from xg_boost import run_xg_boost
from pca import run_pca
from voting import run_voting
from data_preprocessor import get_data

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data(True)

	print("\n-------------------------------------\nAccuracies with 5 features:\n-------------------------------------")

	run_decision_tree(x_train, x_test, y_train, y_test)
	run_k_nearest_neighbour(x_train, x_test, y_train, y_test)
	run_logistic_regression(x_train, x_test, y_train, y_test)
	run_naive_bayes(x_train, x_test, y_train, y_test)
	run_neural_network(x_train, x_test, y_train, y_test)
	run_perceptron(x_train, x_test, y_train, y_test)
	run_random_forest(x_train, x_test, y_train, y_test)
	run_svm(x_train, x_test, y_train, y_test)
	run_xg_boost(x_train, x_test, y_train, y_test)

	x_train, x_test, y_train, y_test = get_data()
	print("\n-------------------------------------\nAccuracies with default data:\n-------------------------------------")

	run_decision_tree(x_train, x_test, y_train, y_test)
	run_k_nearest_neighbour(x_train, x_test, y_train, y_test)
	run_logistic_regression(x_train, x_test, y_train, y_test)
	run_naive_bayes(x_train, x_test, y_train, y_test)
	run_neural_network(x_train, x_test, y_train, y_test)
	run_perceptron(x_train, x_test, y_train, y_test)
	run_random_forest(x_train, x_test, y_train, y_test)
	run_svm(x_train, x_test, y_train, y_test)
	run_xg_boost(x_train, x_test, y_train, y_test)

	print("\n-------------------------------------\nAccuracies with dimensionality reduction using PCA:\n-------------------------------------")
	new_x_train, new_x_test = run_pca(x_train, x_test, y_train, y_test, 10)

	run_decision_tree(new_x_train, new_x_test, y_train, y_test)
	run_k_nearest_neighbour(new_x_train, new_x_test, y_train, y_test)
	run_logistic_regression(new_x_train, new_x_test, y_train, y_test)
	run_naive_bayes(new_x_train, new_x_test, y_train, y_test)
	run_neural_network(new_x_train, new_x_test, y_train, y_test)
	run_perceptron(new_x_train, new_x_test, y_train, y_test)
	run_random_forest(new_x_train, new_x_test, y_train, y_test)
	run_svm(new_x_train, new_x_test, y_train, y_test)
	run_xg_boost(new_x_train, new_x_test, y_train, y_test)

	print("\n-------------------------------------\nAccuracy with Voting:\n-------------------------------------")
	run_voting(new_x_train, new_x_test, y_train, y_test)