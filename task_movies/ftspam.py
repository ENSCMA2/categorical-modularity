import csv
import numpy as np
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer
from google_trans_new import google_translator
import random
import fasttext
import sys

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian']

def load_data(sheet, fac, indices):
    x = []
    with open((sheet), "r+") as o:
      for k in range(10000):
        l = str(o.readline()).strip("[] \n").split(",")
        lf = [float(i) for i in l]
        print(len(lf))
        x.append(lf)
    x = np.array(x)
    y = np.array(['positive'] * 5000 + ['negative'] * 5000)
    X_TRAIN = np.array([x[j] for j in range(len(x)) if j not in indices])
    Y_TRAIN = np.array([y[j] for j in range(len(y)) if j not in indices])
    X_TEST = np.array([x[j] for j in range(len(x)) if j in indices])
    Y_TEST = np.array([y[j] for j in range(len(y)) if j in indices])
    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

def vectorize(data, model):
  vecd = []
  for point in data:
    conv = model[point]
    vecd.append(conv)
  return np.array(vecd)

def average(numbers):
  return sum(numbers)/len(numbers)

factor = 1
numtrials = 30

storage = {}
for lang in langs:
  storage[lang] = {"accuracy": [], "precision": [], "recall": [], "f1": []}

for i in range(numtrials):
  print(i)
  indices = random.sample([num for num in range(0, 10000)], 2000)
  for lang in langs:
    print(lang)
    train_data, test_data, train_labels, test_labels = load_data(lang + '_ftmovievecs.txt', factor, indices)

    classifier = LinearSVC()
    classifier.fit(train_data, train_labels)

    predictions = classifier.predict(test_data)

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, pos_label = 'positive')
    recall = recall_score(test_labels, predictions, pos_label = 'positive')
    f1 = f1_score(test_labels, predictions, pos_label = 'positive')
    storage[lang]["accuracy"].append(accuracy)
    storage[lang]["precision"].append(precision)
    storage[lang]["recall"].append(recall)
    storage[lang]["f1"].append(f1)

for lang in langs:
  with open(lang + "spamftmetrics" + str(numtrials) + ".txt", "w") as o:
    o.write(str(average(storage[lang]["accuracy"])) + "\n")
    o.write(str(average(storage[lang]["precision"])) + "\n")
    o.write(str(average(storage[lang]["recall"])) + "\n")
    o.write(str(average(storage[lang]["f1"])) + "\n")
