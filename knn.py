#!/usr/bin/env python
from __future__ import division
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, svm
from sklearn.decomposition import TruncatedSVD
from gensim.parsing.porter import PorterStemmer
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
    train_data = train_data[0:1000]

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_data["Category"], test_size=0.33, shuffle = True)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y = le.transform(y_train)

    le.fit(y_test)
    y2 = le.transform(y_test)

    p = PorterStemmer()
    train_data["Content"] = train_data["Content"] + 4*(" "+train_data['Title'])
    train_data["Content"] = p.stem_documents(train_data["Content"])

    count_vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = count_vectorizer.fit_transform(X_train["Content"])
    X2 = count_vectorizer.fit_transform(X_test["Content"])

    lsa = TruncatedSVD(n_components=30)
    X = lsa.fit_transform(X)

    lsa = TruncatedSVD(n_components=30)
    X2 = lsa.fit_transform(X2)

    return X,X2,y,y2

# Calculate euclidean distance
def euclideanDistance(instance1, instance2, j):
    distance = 0.0
    for i in range(0,j):  # j for components
        distance += pow(instance1[i]-instance2[i],2)
    result = math.sqrt(distance)
    return result

def getNeighbours(X, testInstance, y, k):
	distances = []
	for i in range(0,len(X)):
		dist = euclideanDistance(testInstance, X[i], len(testInstance))  # j = length of row
		distances.append([dist, y[i]])

	distances.sort(key = itemgetter(0))
	neighbours = []
	for i in range(k):
		neighbours.append(distances[i][1])
	return neighbours

def majorityvote(neighbours, X, y, k):
    cnt = Counter(neighbours[0])
    return cnt.most_common(1)[0][0]

# nearest_neighbour --> main
def nearest_neighbour(k=6):
    X, X2, y, y2, = loadData_Preprocess()

    neighbours = []

    success = 0;
    failure = 0;

    for i in range(0,len(X2)):    # for every testInstance
        neighbours = []
        neighbours.append(getNeighbours(X, X2[i], y, k))    # returns k list items with 2 components
        mv = majorityvote(neighbours, X, y, k) # the answer
        if mv == y2[i]:
            success += 1
        else:
            failure += 1

    var = success/len(y2)
    print "success rate = ", var
    return var

nearest_neighbour()
