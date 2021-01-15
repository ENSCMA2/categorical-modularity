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

lang_codes = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

i = int(sys.argv[1])
translator = google_translator()

with open("IMDBselected.tsv", encoding = "ISO-8859-1", mode='r+') as o:
  print(langs[i])
  lines = csv.reader(o, delimiter = '\t')
  data = list(lines)
  x = [point[0] for point in data]
  print(len(x))
  j = 0
  with open("IMDB" + langs[i] + ".txt", "w") as n:
    for point in x:
      t = translator.translate(point, lang_src = 'en', lang_tgt = lang_codes[i])
      print(j)
      if type(t) != str:
        t = t[0]
      n.write(t + "\n")
      j += 1
