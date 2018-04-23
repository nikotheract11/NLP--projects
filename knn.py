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

    le = preprocessing.LabelEncoder()
    le.fit(train_data["Category"])
    y = le.transform(train_data["Category"])
    #set(y)

    count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = count_vectorizer.fit_transform(train_data["Category"])
    X = list(X)
    return X,y,train_data

# Calculate euclidean distance
def euclideanDistance(instance1, instance2):    # testInstance1 is number
    distance = 0
    #for x in range(4):  # 5 for components
    print "instance1 = ", instance2[0][0][0]
    distance += pow(instance1-instance2[4],2)
    result = math.sqrt(distance)
    return result

def getNeighbours(X, testInstance, k):
	distances = []
	for x in X:
		dist = euclideanDistance(testInstance, x)
		distances.append([x, dist])
	distances.sort(key = itemgetter(1))
	neighbours = []
	for x in range(k):
		neighbours.append(distances[x][0])
	return neighbours

# nearest_neighbour --> main
def nearest_neighbour():
    X ,y, train_data = loadData_Preprocess()
    print "X = ", X
    print "y = ", y

    neighbours = []

    for x in range(len(y)):
        #print "x = ", x
        neighbours.append(getNeighbours(X,y[x],3))

nearest_neighbour()
