import os
import math
import random
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, plot_confusion_matrix
from nltk.corpus import reuters
from sklearn.model_selection import train_test_split
from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from scipy.sparse import csr_matrix
import scipy
from scipy.io import arff
import time

def average(numbers):
  return sum(numbers)/len(numbers)

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

for col in ["2", "3", "4"]:
  cats = []
  with open("categories" + col + ".txt", "r") as o:
    for i in range(500):
      cats.append(str(o.readline()).strip("\n"))
  categories = list(set(cats))
  categories = [cat for cat in categories]
  storage = {}
  for lang in langs:
    storage[lang] = {"accuracy": [],
                     "precision": [],
                     "recall": [],
                     "f1": [],
                     "macro_precision": [],
                     "macro_recall": [],
                     "macro_f1": []}

  numtrials = 0
  while numtrials < 30:
    test_indices = {}
    for i in range(100):
      r = random.randint(0, 500)
      while str(r) in test_indices.keys():
        r = random.randint(0, 500)
      test_indices[str(r)] = "test"
    for langnum in range(len(langs)):
      lang = langs[langnum]
      allvecs = []
      with open(lang + "musevecs.txt", "r") as o:
        for i in range(500):
          l = str(o.readline()).strip("[] \n").split(",")
          lf = [float(i) for i in l]
          allvecs.append(lf)
      train = []
      test = []
      trainl = []
      testl = []
      for i in range(500):
        try:
          check = test_indices[str(i)]
          test.append(allvecs[i])
          testl.append(cats[i])
        except:
          train.append(allvecs[i])
          trainl.append(cats[i])
      vectorised_train_data = np.array(train)
      vectorised_test_data = np.array(test)
      train_labels, test_labels = np.array(trainl), np.array(testl)
      classifier = LinearSVC()
      try:
        classifier.fit(vectorised_train_data, train_labels)
      except:
        for oops in range(langnum):
          del storage[langs[oops]]["accuracy"][-1]
          del storage[langs[oops]]["precision"][-1]
          del storage[langs[oops]]["recall"][-1]
          del storage[langs[oops]]["f1"][-1]
          del storage[langs[oops]]["macro_precision"][-1]
          del storage[langs[oops]]["macro_recall"][-1]
          del storage[langs[oops]]["macro_f1"][-1]
        break
      predictions_final = classifier.predict(vectorised_test_data)
      accuracy = accuracy_score(test_labels, predictions_final)
      precision = precision_score(test_labels, predictions_final, average='micro')
      recall = recall_score(test_labels, predictions_final, average='micro')
      f1 = f1_score(test_labels, predictions_final, average='micro')
      accuracy = accuracy_score(test_labels, predictions_final)
      precision_m = precision_score(test_labels, predictions_final, average='macro')
      recall_m = recall_score(test_labels, predictions_final, average='macro')
      f1_m = f1_score(test_labels, predictions_final, average='macro')
      storage[lang]["accuracy"].append(accuracy)
      storage[lang]["precision"].append(precision)
      storage[lang]["recall"].append(recall)
      storage[lang]["f1"].append(f1)
      storage[lang]["macro_precision"].append(precision_m)
      storage[lang]["macro_recall"].append(recall_m)
      storage[lang]["macro_f1"].append(f1_m)
    numtrials += 1
  for lang in langs:
    with open(lang + "svmMUSE" + col + "metrics.txt", "w") as o:
      o.write(str(average(storage[lang]["accuracy"])) + "\n")
      o.write(str(average(storage[lang]["precision"])) + "\n")
      o.write(str(average(storage[lang]["recall"])) + "\n")
      o.write(str(average(storage[lang]["f1"])) + "\n")
      o.write(str(average(storage[lang]["macro_precision"])) + "\n")
      o.write(str(average(storage[lang]["macro_recall"])) + "\n")
      o.write(str(average(storage[lang]["macro_f1"])) + "\n")
