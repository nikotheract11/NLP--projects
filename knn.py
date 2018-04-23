#!/usr/bin/env python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os import path

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from operator import itemgetter

import math

# load the train_set, preprocess it and return its train and test sets
def loadData_Preprocess():
    currdir = path.dirname(__file__)

    train_data = pd.read_csv('./datasets/train_set.csv', sep="\t")	# './datasets/train_set.csv'
    train_data = train_data[0:25]

    le = preprocessing.LabelEncoder()
    le.fit(train_data["Category"])
    y = le.transform(train_data["Category"])
    #set(y)

    count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = count_vectorizer.fit_transform(train_data["Content"])
    #X = list(X)
    X = X.toarray()
    return X,y,train_data

# Calculate euclidean distance
def euclideanDistance(instance1, instance2, j):    # testInstance1 is number, instance2 is not
    distance = 0
    for i in range(j):  # j for components
        distance += pow(instance1-instance2[i],2)
    result = math.sqrt(distance)
    return result

def getNeighbours(X, testInstance, k):
	distances = [], j = 10
	for x in X:
		dist = euclideanDistance(testInstance, x, j)  # j = length of row
		distances.append([x, dist])
	distances.sort(key = itemgetter(1))
	neighbours = []
	for x in range(k):
		neighbours.append(distances[x])
	return neighbours

# nearest_neighbour --> main
def nearest_neighbour():
    X ,y, train_data = loadData_Preprocess()
    print "X = ", X
    print "y = ", y

    neighbours = []

    for yi in y:
        neighbours.append(getNeighbours(X,yi,3))

    print neighbours

nearest_neighbour()
