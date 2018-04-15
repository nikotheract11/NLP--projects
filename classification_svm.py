#!/usr/bin/env python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing, svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os import path

from sklearn import svm

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

clf = svm.SVC()
clf.fit(X,y)

y_pred = clf.predict(X)
y_pred

predicted_categories = le.inverse_transform(y_pred)
print predicted_categories

print classification_report(y, y_pred, target_names=list(le.classes_))
