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

currdir = path.dirname(__file__)

train_data = pd.read_csv('./datasets/train_set.csv', sep="\t")	# './datasets/train_set.csv'
#train_data = train_data[0:25]

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])			# "Category"

y = le.transform(train_data["Category"])	# "Category"
set(y)
print y

count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X = count_vectorizer.fit_transform(train_data["Category"])

print X		# vector of all columns in identifiers

# SUPPORT VECTOR MACHINE
#clf = svm.SVC()

# FOREST CLASSIFIER
#clf = RandomForestClassifier()

# Naive Bayes
clf = MultinomialNB()

clf.fit(X,y)

y_pred = clf.predict(X)
y_pred

predicted_categories = le.inverse_transform(y_pred)
print predicted_categories

print classification_report(y, y_pred, target_names=list(le.classes_))

######################## CROSS VALIDATION ####################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)	# split set

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape


scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']

scores = cross_validate(clf, X, y, cv=10, scoring=scoring)

sorted(scores.keys())

# k-fold
print "10-Fold..."
kf = KFold(n_splits=10)
for train, test in kf.split(X):		# train set kai test set split from original
	print "%s %s" % (train, test)

# precision, recall, f-measure, accuracy
print "precision, recall, f-measure, accuracy ..."
print scores
