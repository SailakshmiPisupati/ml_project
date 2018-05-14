import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

def get_labels():
	# return ["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx","label"]
	return ["idderived_data","stroke_id","avg_x","avg_y","avg_pressure","avg_size","median_x","median_y","median_pressure","median_size","std_dev_x","std_dev_y","std_dev_pressure","std_dev_size","percent25_x","percent50_x","percent75_x","percent100_x","percent25_y","percent50_y","percent75_y","percent100_y","percent25_pressure","percent50_pressure","percent75_pressure","percent100_pressure","percent25_size","percent50_size","percent75_size","percent100_size","userid","mode_x","variance_x","mode_y","variance_y","mode_pressure","mode_size","variance_pressure","variance_size","curvature","slope","stroke_length","gesture_type","hand","auth_user"]

def get_data(top_features = False, test_size=0.2, random_state=0,feature_considered=1,individual_value= False):
	with open('../derived_data_vertical.csv') as voicefile:
	# with open('../voice.csv') as voicefile:
		myFileReader = csv.reader(voicefile, delimiter=',', quotechar='|')
		X = []
		Y = []
		
		for row in myFileReader:
			X.append(row)

		print("X is: ",X[0])

		# preparing attribute and label data
		# for i in range(0,41):
		# 	Y.append(X[i][41][0:-1])
		# 	del(X[i][41])
		for i in range(len(X)):
			Y.append(X[i][44][0:-1])
			del(X[i][44])

		# print("Y",Y)
		# removing column names
		del(X[0]) 
		del(Y[0])

		if top_features and individual_value:
			new_X = []
			for row in X:
				new_row = []
				# print(row[feature_considered])
				new_row.append(row[feature_considered])
				# new_row.append(row[40])
				# new_row.append(row[4])
				# new_row.append(row[22])
				# new_row.append(row[23])
				new_X.append(new_row)
			X = new_X

		if top_features and not individual_value:
			new_X = []
			for row in X:
				new_row = []
				new_row.append(row[11])
				new_row.append(row[13])
				new_row.append(row[32])
				new_row.append(row[38])
				new_row.append(row[39])
				new_X.append(new_row)
			X = new_X


		X = preprocessing.scale(X) # attribute scaling so that attributes with high values do not dominate

		result = train_test_split(X, Y, test_size=test_size, random_state=random_state)
		return list(map(lambda r: np.array(r), result))


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data() # True
	
	print("\nx_train:\n")
	print(len(x_train[0]))
	print("\nx_test:\n")
	print(x_test)
	print("\ny_train:\n")
	print(y_train)
	print("\ny_test:\n")
	print(y_test)