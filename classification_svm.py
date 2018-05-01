#!/usr/bin/env python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, svm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from os import path

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

currdir = path.dirname(__file__)

train_data = pd.read_csv('./datasets/train_set.csv', sep="\t")	# './datasets/train_set.csv'
train_data = train_data[0:25]

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])			# "Category"

y = le.transform(train_data["Category"])	# "Category"
#set(y)
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

############################## testSet_categories.csv

clf = MultinomialNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print "ypred = ", y_pred
predicted_categories = le.inverse_transform(y_pred)
print "predicted_categories = ", predicted_categories

data = {'ID' : [i for i in y_pred],
	'Predicted_Category' : [i for i in predicted_categories]}

df = pd.DataFrame(data, columns = ['ID','Predicted_Category'])

df.to_csv('./testSet_categories.csv', sep='\t')

##############################

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']

scores = cross_validate(clf, X, y, scoring=scoring)

sorted(scores.keys())

# k-fold
print "10-Fold..."
kf = KFold(n_splits=10)
for train, test in kf.split(X):		# train set kai test set split from original
	print "%s %s" % (train, test)

############## EvaluationMetric_10fold.csv

clfnb = MultinomialNB()
clfnb.fit(X,y)
clfrf = RandomForestClassifier()
clfrf.fit(X,y)
clfsvm = svm.SVC()
clfsvm.fit(X,y)
clfknn = KNeighborsClassifier()
clfknn.fit(X,y)

#score = sorted(cross_validate(clfnb, X, y, scoring='accuracy', cv=10, return_train_score=True)).keys()
#print "score = ", score

data = {'Statistic Measure' : ['Accuracy','Precision','Recall','F-Measure'],
	'Naive Bayes' : [math.fsum(cross_validate(clfnb, X, y, scoring='accuracy')['train_score'])/3,math.fsum(cross_validate(clfnb, X, y, scoring='precision_macro')['train_score'])/3,math.fsum(cross_validate(clfnb, X, y, scoring='recall_macro')['train_score'])/3,math.fsum(cross_validate(clfnb, X, y, scoring='f1_macro')['train_score'])/3],
	'Random Forest' : [math.fsum(cross_validate(clfrf, X, y, scoring='accuracy')['train_score'])/3,math.fsum(cross_validate(clfrf, X, y, scoring='precision_macro')['train_score'])/3,math.fsum(cross_validate(clfrf, X, y, scoring='recall_macro')['train_score'])/3,math.fsum(cross_validate(clfrf, X, y,scoring='f1_macro')['train_score'])/3],
	'SVM' : [math.fsum(cross_validate(clfsvm, X, y, scoring='accuracy')['train_score'])/3,math.fsum(cross_validate(clfsvm, X, y, scoring='precision_macro')['train_score'])/3,math.fsum(cross_validate(clfsvm, X, y, scoring='recall_macro')['train_score'])/3,math.fsum(cross_validate(clfsvm, X, y, scoring='f1_macro')['train_score'])/3],
	'KNN' : [math.fsum(cross_validate(clfknn, X, y, scoring='accuracy')['train_score'])/3,math.fsum(cross_validate(clfknn, X, y, scoring='precision_macro')['train_score'])/3,math.fsum(cross_validate(clfknn, X, y, scoring='recall_macro')['train_score'])/3,math.fsum(cross_validate(clfknn, X, y, scoring='f1_macro')['train_score'])/3]}

df = pd.DataFrame(data, columns = ['Statistic Measure','Naive Bayes','Random Forest','SVM','KNN'])

df.to_csv('./EvaluationMetric_10fold.csv', sep='\t')

##############

# precision, recall, f-measure, accuracy
print "precision, recall, f-measure, accuracy ..."
#print scores
