import scipy
from sklearn.neighbors import NearestNeighbors
import numpy as np
import csv

#This program will use K Nearest Neighbors as a machine learning algorithm that predicts
#survival on the Titanic based on a number of characterics about every passenger. 

Xtrain=[] #Will contain the characteristic data for each passenger
Ytrain=[] #WIll contain a binary label for whether each passenger survived

#First job is to read in the data from the training data that Kaggle provides. This
#training data is in the form of a csv file. This CSV file should be in the same
#directory as this program. 

train_file = open('train.csv')
csv_file = csv.reader(trainFile)
for row in train_file:
	#add info to xtrain
	#add info to ytrain


Xtest=[] 
Ytest=[]
test_file = open('test.csv')
for row in test_file:
	#add info to xtest
	#add info to ytest
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
