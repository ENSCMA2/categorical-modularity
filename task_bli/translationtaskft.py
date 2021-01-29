import fasttext
import sys
import io
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix

def load(sheet, l):
  x = []
  with open((sheet), "r+") as o:
    for k in range(l):
      l = str(o.readline()).strip("[] \n").split(",")
      lf = [float(i) for i in l]
      x.append(lf)
  x = np.array(x)
  return x

def loss(pred, true):
  sumdist = 0
  for j in range(len(pred)):
    dist = 0
    for k in range(len(pred[j])):
      dist += (pred[j][k] - true[j][k]) * (pred[j][k] - true[j][k])
    sumdist += dist ** 0.5
  return sumdist/len(pred)

def closs(pred, true):
  total = 0
  for j in range(len(pred)):
    dp = 0
    mag1 = 0
    mag2 = 0
    for k in range(len(pred[j])):
      dp += pred[j][k] * true[j][k]
      mag1 += pred[j][k] * pred[j][k]
      mag2 += true[j][k] * true[j][k]
    if mag1 > 0 and mag2 > 0:
      total += dp / ((mag1 ** 0.5) * (mag2 ** 0.5))
  return total / len(pred)

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']

i = int(sys.argv[1])
lang = langs[i]
print(lang)

from_train_x = load(lang + "from_english_en_translationftvecstrain.txt", 5000)
from_train_y = load(lang + "from_english_" + lang + "_translationftvecstrain.txt", 5000)
from_test_x = load(lang + "from_english_en_translationftvecstest.txt", 1500)
from_test_y = load(lang + "from_english_" + lang + "_translationftvecstest.txt", 1500)

to_train_y = load(lang + "to_english_en_translationftvecstrain.txt", 5000)
to_train_x = load(lang + "to_english_" + lang + "_translationftvecstrain.txt", 5000)
to_test_y = load(lang + "to_english_en_translationftvecstest.txt", 1500)
to_test_x = load(lang + "to_english_" + lang + "_translationftvecstest.txt", 1500)

from_model = LinearRegression()
from_model.fit(from_train_x, from_train_y)
from_predictions = from_model.predict(from_test_x)
from_loss = loss(from_predictions, from_test_y)
from_closs = closs(from_predictions, from_test_y)
print("from")
print([from_loss, from_closs])


to_model = LinearRegression()
to_model.fit(to_train_x, to_train_y)
to_predictions = to_model.predict(to_test_x)
to_loss = loss(to_predictions, to_test_y)
to_closs = closs(to_predictions, to_test_y)
print("to")
print([to_loss, to_closs])




