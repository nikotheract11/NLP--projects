#!/usr/bin/env python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing, svm

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from os import path
from wordcloud import WordCloud,ImageColorGenerator

from sklearn import svm

currdir = path.dirname(__file__)

alice_mask = np.array(Image.open(path.join(currdir, "drop.png")))


train_data = pd.read_csv('./datasets/train_set.csv', sep="\t")
sw = set(ENGLISH_STOP_WORDS)
sw.add("say")
sw.add("says")
sw.add("said")
sw.add("new")
categories = set(train_data["Category"])
for cat in categories:
	text=train_data[train_data["Category"]==cat]
	text=text["Content"]
	text=text.to_string()

	wc = WordCloud(background_color="white", mask=alice_mask,
	stopwords=sw,random_state=18,colormap="plasma",max_font_size=40,max_words=2000)
	wdcd = wc.generate(text)

	wc.to_file(path.join(currdir, cat + ".png"))
