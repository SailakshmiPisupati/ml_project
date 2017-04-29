import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

def get_data(test_size=0.2, random_state=0):
	with open('../voice.csv') as voicefile:
		myFileReader = csv.reader(voicefile, delimiter=',', quotechar='|')
		X = []
		Y = []
		
		for row in myFileReader:
			X.append(row)

		# preparing attribute and label data
		for i in range(len(X)):
			Y.append(X[i][20][1:-1])
			del(X[i][20])

		# removing column names
		del(X[0]) 
		del(Y[0])

		X = preprocessing.scale(X) # attribute scaling so that attributes with high values do not dominate

		result = train_test_split(X, Y, test_size=test_size, random_state=random_state)
		return list(map(lambda r: np.array(r), result))

if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	
	# print("\nx_train:\n")
	# print(x_train)
	# print("\nx_test:\n")
	# print(x_test)
	print("\ny_train:\n")
	print(y_train)
	# print("\ny_test:\n")
	# print(y_test)