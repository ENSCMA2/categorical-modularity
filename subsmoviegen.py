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
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import stopwords
import io

i = int(sys.argv[1])

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']

model = fasttext.load_model('subs2vec/subs.' + language_paths[i] + '.bin')

with open("IMDB" + langs[i] + ".txt", "r+") as o:
  print(langs[i])
  text = [line.lower().strip("\n") for line in o]
  with open(langs[i] + "_subsmovievecs.txt", "w") as n:
    for j in range(len(text)):
      print(j)
      vec = list(model[text[j]])
      print(len(vec))
      n.write(str(vec) + "\n")
