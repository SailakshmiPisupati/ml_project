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

feature_dict = ["idderived_data","stroke_id","avg_x","avg_y","avg_pressure","avg_size","median_x","median_y","median_pressure","median_size","std_dev_x","std_dev_y","std_dev_pressure","std_dev_size","percent25_x","percent50_x","percent75_x","percent100_x","percent25_y","percent50_y","percent75_y","percent100_y","percent25_pressure","percent50_pressure","percent75_pressure","percent100_pressure","percent25_size","percent50_size","percent75_size","percent100_size","userid","mode_x","variance_x","mode_y","variance_y","mode_pressure","mode_size","variance_pressure","variance_size","curvature","slope","stroke_length","gesture_type","hand","auth_user"]

def deleteFileContent(fName):
    with open(fName, "w"):
        pass

if __name__ == '__main__':
	deleteFileContent("output.txt")
	for i in range(2,44):
		x_train, x_test, y_train, y_test = get_data(top_features= True,feature_considered = i, individual_value = True)

		print("\n-------------------------------------\nAccuracies with features:",feature_dict[i],"\n-------------------------------------")
		with open("output.txt", "a") as text_file:
			print(f"\n-------------------------------------\nAccuracies with features:",feature_dict[i],"\n-------------------------------------", file=text_file)
		run_decision_tree(x_train, x_test, y_train, y_test)
		run_k_nearest_neighbour(x_train, x_test, y_train, y_test)
		run_logistic_regression(x_train, x_test, y_train, y_test)
		run_naive_bayes(x_train, x_test, y_train, y_test)
		run_neural_network(x_train, x_test, y_train, y_test)
		run_perceptron(x_train, x_test, y_train, y_test)
		run_random_forest(x_train, x_test, y_train, y_test)
		run_svm(x_train, x_test, y_train, y_test)
		run_xg_boost(x_train, x_test, y_train, y_test)

	print("\n-------------------------------------\nAccuracy with Voting in top 5 features:\n-------------------------------------")
	with open("output.txt", "a") as text_file:
		print(f"\n-------------------------------------\nAccuracy with Voting in top 5 features:\n-------------------------------------", file=text_file)
	
	run_voting(x_train, x_test, y_train, y_test)


	print("\n-------------------------------------\nAccuracies with top 5 features:\n-------------------------------------")
	with open("output.txt", "a") as text_file:
			print(f"\n-------------------------------------\nAccuracy with top 5 features:\n-------------------------------------", file=text_file)
	
	run_decision_tree(x_train, x_test, y_train, y_test)
	run_k_nearest_neighbour(x_train, x_test, y_train, y_test)
	run_logistic_regression(x_train, x_test, y_train, y_test)
	run_naive_bayes(x_train, x_test, y_train, y_test)
	run_neural_network(x_train, x_test, y_train, y_test)
	run_perceptron(x_train, x_test, y_train, y_test)
	run_random_forest(x_train, x_test, y_train, y_test)
	run_svm(x_train, x_test, y_train, y_test)
	run_xg_boost(x_train, x_test, y_train, y_test)

	print("\n-------------------------------------\nAccuracy with Voting in top 5 features:\n-------------------------------------")
	with open("output.txt", "a") as text_file:
		print(f"\n-------------------------------------\nAccuracy with Voting in top 5 features:\n-------------------------------------", file=text_file)
	
	run_voting(x_train, x_test, y_train, y_test)

	x_train, x_test, y_train, y_test = get_data()
	print("\n-------------------------------------\nAccuracies with all 22 features:\n-------------------------------------")
	with open("output.txt", "a") as text_file:
		print(f"\n-------------------------------------\nAccuracy with with all 22 features:\n-------------------------------------", file=text_file)
	

	run_decision_tree(x_train, x_test, y_train, y_test)
	run_k_nearest_neighbour(x_train, x_test, y_train, y_test)
	run_logistic_regression(x_train, x_test, y_train, y_test)
	run_naive_bayes(x_train, x_test, y_train, y_test)
	run_neural_network(x_train, x_test, y_train, y_test)
	run_perceptron(x_train, x_test, y_train, y_test)
	run_random_forest(x_train, x_test, y_train, y_test)
	run_svm(x_train, x_test, y_train, y_test)
	run_xg_boost(x_train, x_test, y_train, y_test)

	print("\n-------------------------------------\nAccuracy with Voting in all 22 features:\n-------------------------------------")
	with open("output.txt", "a") as text_file:
		print(f"\n-------------------------------------\nAccuracy with with all 22 features:\n-------------------------------------", file=text_file)
	
	run_voting(x_train, x_test, y_train, y_test)

	print("\n-------------------------------------\nAccuracies with dimensionality reduction using PCA (5 components):\n-------------------------------------")
	with open("output.txt", "a") as text_file:
		print(f"\n-------------------------------------\nAccuracies with dimensionality reduction using PCA (5 components):\n-------------------------------------", file=text_file)
	
	new_x_train, new_x_test = run_pca(x_train, x_test, y_train, y_test, 5)

	run_decision_tree(new_x_train, new_x_test, y_train, y_test)
	run_k_nearest_neighbour(new_x_train, new_x_test, y_train, y_test)
	run_logistic_regression(new_x_train, new_x_test, y_train, y_test)
	run_naive_bayes(new_x_train, new_x_test, y_train, y_test)
	run_neural_network(new_x_train, new_x_test, y_train, y_test)
	run_perceptron(new_x_train, new_x_test, y_train, y_test)
	run_random_forest(new_x_train, new_x_test, y_train, y_test)
	run_svm(new_x_train, new_x_test, y_train, y_test)
	run_xg_boost(new_x_train, new_x_test, y_train, y_test)

	print("\n-------------------------------------\nAccuracy with Voting along with PCA:\n-------------------------------------")
	with open("output.txt", "a") as text_file:
		print(f"\n-------------------------------------\nAccuracy with Voting along with PCA::\n-------------------------------------", file=text_file)
	
	run_voting(new_x_train, new_x_test, y_train, y_test)
