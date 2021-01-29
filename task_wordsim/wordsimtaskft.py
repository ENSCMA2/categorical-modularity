import fasttext
import sys
from google_trans_new import google_translator
import numpy as np
import io
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import stopwords
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']

datafiles = [langs[i] + "_wordsim_ftvecs.txt" for i in range(len(langs))]
labelfiles = [langs[i] + "_wordsim_ftscores.txt" for i in range(len(langs))]

numtrials = 30
possibilities = [i for i in range(500)]
losses = [0] * len(langs)

for i in range(numtrials):
  print(i)
  test = random.sample(possibilities, 100)
  for j in range(len(langs)):
    print(langs[j])
    traindata = []
    testdata = []
    trainlabels = []
    testlabels = []
    with open(datafiles[j], "r+") as o:
      for k in range(500):
        l = [float(y) for y in str(o.readline()).strip("[] \n").split(",")]
        if k in test:
          testdata.append(l)
        else:
          traindata.append(l)
    with open(labelfiles[j], "r+") as o:
      for k in range(500):
        s = float(str(o.readline()))
        if k in test:
          testlabels.append(s)
        else:
          trainlabels.append(s)
    traindata = np.array(traindata)
    testdata = np.array(testdata)
    trainlabels = np.array(trainlabels)
    testlabels = np.array(testlabels)
    model = LinearRegression()
    model.fit(traindata, trainlabels)
    preds = model.predict(testdata)
    loss = 0
    for num in range(len(preds)):
      loss += (preds[num] - testlabels[num]) * (preds[num] - testlabels[num])
    loss = loss / len(preds)
    losses[j] += loss / numtrials

for l in losses:
  print(l)







