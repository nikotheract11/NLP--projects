#!/usr/bin/env python
from __future__ import division
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
from collections import Counter # used for most appearances of value in list

# load the train_set, preprocess it and return its train and test sets
def loadData_Preprocess():
    currdir = path.dirname(__file__)

    train_data = pd.read_csv('./datasets/train_set.csv', sep="\t")	# './datasets/train_set.csv'
    train_data = train_data[0:25]

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_data["Category"], test_size=0.33, random_state=42)
    print X_test, y_test

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y = le.transform(y_train)

    le2 = preprocessing.LabelEncoder()
    le2.fit(y_test)
    y2 = le2.transform(y_test)

    count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = count_vectorizer.fit_transform(X_train["Content"])
    X2 = count_vectorizer.fit_transform(X_test["Content"])

    X = X.toarray()
    X2 = X2.toarray()

    return X,X2,y,y2,train_data,le

# Calculate euclidean distance
def euclideanDistance(instance1, instance2, j):    # testInstance1 is number, instance2 is not
    distance = 0
    for i in range(0,1):  # j for components
        distance += pow(instance2[i]-instance1[i],2)
    result = math.sqrt(distance)
    return result

def getNeighbours(X, testInstance, k):
	distances = []; j = len(X[0])
	for i in range(0,len(X)):
		dist = euclideanDistance(testInstance, X[i], j)  # j = length of row
		distances.append([X[i], dist, i])

	distances.sort(key = itemgetter(1))
	neighbours = []
	for i in range(k):
		neighbours.append(distances[i])
        #print "i = ", distances[i]
	return neighbours

def majorityvote(neighbours, X, le, y, k):
    categories = []
    s = []
    for i in range(k):
        categories.append(y[neighbours[0][i][2]])
    #print "categories = ", categories
    #s = le.inverse_transform(categories)    # contains all categories
    #print "s = ", s
    cnt = Counter(categories)
    print cnt
    return cnt.most_common(1)[0][0]

# nearest_neighbour --> main
def nearest_neighbour():
    X, X2, y, y2, train_data, le = loadData_Preprocess()
    print "X = ", X
    print "X2 = ", X2
    print "y = ", y
    print "len(y2) = ", len(y2)
    print "len(X2) = ", len(X2)

    neighbours = []
    k = 2

    success = 0;
    failure = 0;

    for i in range(0,len(X2)):    # for every testInstance
        neighbours = []
        neighbours.append(getNeighbours(X,X2[i],k))    # returns k list items with 2 components
        ela = majorityvote(neighbours, X, le, y, k) # the answer
        print ela, " ", y2[i]
        if ela == y2[i]:
            success += 1
        else:
            failure += 1

    var = success/len(y2)
    print "success rate = ", var

nearest_neighbour()
