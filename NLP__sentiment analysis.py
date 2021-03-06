#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 01:48:15 2021

@author: rolvy-dicken
"""


# importing librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting = 3)

# Cleaning the data
import re 
import nltk
nltk.download("stopwords")   
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ',  dataset["Review"][i])
    review = review.lower()
    review = review.split()
    ps =    PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1565)
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values

# Spliting the dataset into the training set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state=0)

# Fitting Naive Bayes to the training set 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)      

# predicting the test set results
y_pred = clf.predict(X_test)           

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
