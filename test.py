#!/usr/bin/env python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pandas as pd


from os import path
from wordcloud import WordCloud

train_data = pd.read_csv('/home/nikos/Downloads/datasets/train_set.csv', sep="\t")
train_data = train_data[0:25]

categories = set(train_data["Category"])
for cat in categories:
    text=train_data[train_data["Category"]==cat]
    text=text["Content"]
    text=text.to_string()
    wordcloud = WordCloud(stopwords=ENGLISH_STOP_WORDS).generate(text)

    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
