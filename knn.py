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
from collections import Counter # used for most appearances of value in list

# load the train_set, preprocess it and return its train and test sets
def loadData_Preprocess():
    currdir = path.dirname(__file__)

    train_data = pd.read_csv('./datasets/train_set.csv', sep="\t")	# './datasets/train_set.csv'
    train_data = train_data[0:50]

    le = preprocessing.LabelEncoder()
    le.fit(train_data["Category"])
    y = le.transform(train_data["Category"])
    #set(y)

    count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = count_vectorizer.fit_transform(train_data["Content"])
    #X = list(X)
    X = X.toarray()
    return X,y,train_data,le

# Calculate euclidean distance
def euclideanDistance(instance1, instance2, j):    # testInstance1 is number, instance2 is not
    distance = 0
    for i in range(j):  # j for components
        distance += pow(instance1-instance2[i],2)
    result = math.sqrt(distance)
    return result

def getNeighbours(X, testInstance, k):
	distances = []; j = len(X[0])
	for x in X:
		dist = euclideanDistance(testInstance, x, j)  # j = length of row
		distances.append([x, dist, testInstance])

	distances.sort(key = itemgetter(1))
	neighbours = []
	for i in range(k):
		neighbours.append(distances[i])
        #print "i = ", distances[i]
	neighbours = neighbours[0:k];  return neighbours

def majorityvote(neighbours, X, le, y, k):
    categories = []
    s = []
    for i in range(k):
        categories.append(neighbours[0][i][2])
    #print "categories = ", categories
    s = le.inverse_transform(categories)    # contains all categories
    #print "s = ", s
    cnt = Counter(s)
    return cnt.most_common(1)[0][0]

# nearest_neighbour --> main
def nearest_neighbour():
    X ,y, train_data, le = loadData_Preprocess()
    print "X = ", X
    print "y = ", y

    neighbours = []
    k = 2

    success = 0;
    failure = 0;

    for yi in y:
        neighbours = []
        neighbours.append(getNeighbours(X,yi,k))    # returns k list items with 2 components
        ela = majorityvote(neighbours, X, le, y, k) # the answer
        if ela == yi:
            success += 1
        else:
            failure += 1
        #print "answer is : ", ela

    print "success rate = ", float(success/len(y))

nearest_neighbour()
