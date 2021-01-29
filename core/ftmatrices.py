import io
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import stopwords
import fasttext
import fasttext.util
import sys

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
snowball = SnowballStemmer("porter")
sl = ['arabic', 'english', 'dutch', 'portuguese', 'spanish', 'danish', 'finnish',
'french', 'german', 'hungarian', 'italian', 'norwegian', 'romanian', 'russian',
'swedish']

path_root = '/Users/karinahalevy/Code/MUSE/data/wiki.multi.'
path_suffix = '.vec'
language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi',
                  'fr', 'de', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']
languages = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french', 'german',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']
word_files = [i + '.txt' for i in languages]


def get_nn_words(word, model):
  nns = []
  m = list(model.get_nearest_neighbors(word, k = 1000000))
  print(len(m))
  for s, w in m:
    nns.append(w)
  return nns

i = int(sys.argv[1])
print(languages[i])
mat = []
model = fasttext.load_model('wiki.' + language_paths[i] + '.bin')
with open(word_files[i], "r") as words:
  wds = [line.lower().strip("\n") for line in words]
  print(len(wds))
for j in range(1):
  neighbs = get_nn_words(wds[j], model)
  '''
  vals = []
  for n in range(1):
    try:
      k = neighbs.index(wds[n])
    except:
      k = -1
    vals.append(k)
  mat.append(vals)
  '''
'''
with open(languages[i] + "matrixft500.txt", "w") as o:
  for q in range(len(wds)):
    o.write(str(mat[q]) + "\n")
print("done with" + languages[i])
'''
