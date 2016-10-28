from __future__ import division
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier
from keras import backend as K
import scipy
import numpy as np
import csv
import sys

# This program will use Neural Networks as a machine learning algorithm that predicts
# survival on the Titanic based on a number of characterics about every passenger. 

Xtrain=[] # Will contain the characteristic data for each passenger. Xtrain will 
		  # contain info about the passenger's class, sex, age, siblings, parch,
		  # fare, and cabin, and embark status
Ytrain=[] # WIll contain a binary label for whether each passenger survived
Xtest=[] # Will contain the characteristic data for the passengers in the test set
numTrainExamples = 891
numTestExamples = 418

# The next few functions are going to serve as preprocessing steps for the different
# features found in the input. Preprocessing will mainly end up being mean normalization
# and filling in blank values. The main process function (below) just applies mean 
# normalization to the values in the set. However, we need specific process functions
# for characteristics like gender, where we have to assign a number value to the 
# characteristic of male or female. 

def process(pclass):
	pclass = [float(x) for x in pclass]
	mean = sum(pclass) / float(len(pclass))
	my_range = max(pclass) - min(pclass)
	pclass = [(x-mean)/my_range for x in pclass]
	return pclass

def processGender(gender):
	genderToNum=[]
	for x in gender:
		if (x == "male"):
			genderToNum.append(-.5)
		else: # x is female
			genderToNum.append(.5)
	return genderToNum

def processAgeorFare(age):
	ageWithoutBlanks=[]
	for x in age:
		if (x != ""): 
			ageWithoutBlanks.append(x)
	ageWithoutBlanks = [float(x) for x in ageWithoutBlanks]
	mean = sum(ageWithoutBlanks) / float(len(ageWithoutBlanks))
	my_range = max(ageWithoutBlanks) - min(ageWithoutBlanks)
	#median = np.median(ageWithoutBlanks)
	for x in range(0,len(age)):
		if (age[x] == ""): # If there is a blank value, then just set it to mean
			age[x] = mean
	age = [float(x) for x in age]
	age = [(x-mean)/my_range for x in age]
	return age

def processCabin(cabin):
	cabinToNum=[]
	for x in gender:
		if (x == ""):
			cabinToNum.append(-.5)
		else:
			cabinToNum.append(.5)
	return cabinToNum

def processEmbarked(embarked):
	embarkedToNum=[]
	for x in embarked:
		if (x == "S"):
			embarkedToNum.append(-.5)
		elif (x == "Q"):
			embarkedToNum.append(0.5)
		else: # x is C or blank
			embarkedToNum.append(0)
	return embarkedToNum

# This functions passes X_batch into a trained Keras model and returns the activations
# of a given layer. 
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations

# First job is to read in the data from the training data that Kaggle provides. This
# training data is in the form of a csv file. This CSV file should be in the same
# directory as this program. 

# We want to skip the first row in the csv file because it just 
# contains column header
skip = True
train_file = open('train.csv')
csv_file = csv.reader(train_file)

# Creating temporary lists where we store data for each feature/characteristic
gender,Pclass,age,sibSP,parch,fare,cabin,embarked = ([] for i in range(8))

for row in csv_file:
	if (skip == True):
		skip = False
		continue
	# Filling lists with values from train.csv
	Ytrain.append(row[1]) 
	Pclass.append(row[2]) 
	gender.append(row[4]) 
	age.append(row[5]) 
	sibSP.append(row[6]) 
	parch.append(row[7]) 
	fare.append(row[9]) 
	cabin.append(row[10]) 
	embarked.append(row[11]) 

# Processing each feature list	
Pclass = process(Pclass)
gender = processGender(gender)
age = processAgeorFare(age)
sibSP = process(sibSP)
parch = process(parch)
fare = processAgeorFare(fare)
cabin = processCabin(cabin)
embarked = processEmbarked(embarked)

# Adding values from previous feature lists to one large Xtrain list of lists
for x in range(0,numTrainExamples):
	Xtrain.append([Pclass[x],gender[x],age[x],sibSP[x],
		parch[x],fare[x],cabin[x],embarked[x]])

# Repeating same process for test file (except we don't know Ytest)
skip = True
test_file = open('test.csv')
csv_file2 = csv.reader(test_file)

gender,Pclass,age,sibSP,parch,fare,cabin,embarked = ([] for i in range(8))

for row in csv_file2:
	if (skip == True):
		skip = False
		continue
	Pclass.append(row[1]) 
	gender.append(row[3]) 
	age.append(row[4]) 
	sibSP.append(row[5]) 
	parch.append(row[6])
	fare.append(row[8]) 
	cabin.append(row[9]) 
	embarked.append(row[10]) 

Pclass = process(Pclass)
gender = processGender(gender)
age = processAgeorFare(age)
sibSP = process(sibSP)
parch = process(parch)
fare = processAgeorFare(fare)
cabin = processCabin(cabin)
embarked = processEmbarked(embarked)

for x in range(0,numTestExamples):
	Xtest.append([Pclass[x],gender[x],age[x],sibSP[x],
		parch[x],fare[x],cabin[x],embarked[x]])

# Representing the list of lists as numpy arrays so that we can 
# use the different scikit functions. 
Xtrain = np.asarray(Xtrain)
Xtest = np.asarray(Xtest)
Ytrain = np.asarray(Ytrain)
Ytrain = np_utils.to_categorical(Ytrain, 2)

print Ytrain.shape

# NETWORK ARCHITECTURE
model = Sequential()
model.add(Dense(32, input_dim=8))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.20))
model.add(Dense(48))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('softmax'))

# TRAINING
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, batch_size=32, nb_epoch=1,shuffle=True)

# The idea is that I want a trained neural network for classification, but instead
# of using the class ouputs of the network, I want the activations at a given layer, 
# which I can then feed into a KNN

activations = get_activations(model, 3, Xtest)

# TESTING

neigh = KNeighborsClassifier(n_neighbors=17)
print 'Fitting Nearest Neighbors'
XtrainNewDimensions = get_activations(model,3,Xtrain)
neigh.fit(XtrainNewDimensions, Ytrain)
results = np.ones((numTestExamples,2))
counter = numTrainExamples + 1

print 'Predicting outputs for testing dataset'
for x in range(0,numTestExamples):
	print (neigh.predict(activations[x]))
	results[x,1] = (neigh.predict(activations[x]))[0][0]
	results[x,0] = counter
	counter = counter + 1


#results = np.ones((numTestExamples,2))
#counter = numTrainExamples + 1
#print 'Predicting outputs for testing dataset'
#temp = model.predict_classes( Xtest, batch_size=32, verbose=1)

#for num in range(0,numTestExamples):	
#	results[num,1] = temp[num]
#	results[num,0] = counter
#	counter = counter + 1

#Saving predictions into a test file that can be uploaded to Kaggle
#NOTE: You have to add a header row before submitting the txt file
np.savetxt('result.csv', results, delimiter=',', fmt = '%i') 