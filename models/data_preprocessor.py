import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def get_data(test_size=0.2, random_state=0):
	with open('../voice.csv') as voicefile:
		myFileReader = csv.reader(voicefile, delimiter=',', quotechar='|')
		X = []
		Y = []
		for row in myFileReader:
			X.append(row)

		# preparing attribute and label data
		for i in range(len(X)):
			Y.append(X[i][20])
			del(X[i][20])
		# removing column names
		del(X[0]) 
		del(Y[0])
		X = preprocessing.scale(X) # attribute scaling so that attributes with high values do not dominate
		return train_test_split(X, Y, test_size=test_size, random_state=random_state)


if __name__ == '__main__':
	X_train, X_test, y_train, y_test = get_data()
	
	print("\nX_train:\n")
	print(X_train)
	print("\nX_test:\n")
	print(X_test)
	print("\ny_train:\n")
	print(y_train)
	print("\ny_test:\n")
	print(y_test)