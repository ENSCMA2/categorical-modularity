import fasttext
import sys
from google_trans_new import google_translator
import numpy as np

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']

i = int(sys.argv[1])
print(langs[i])
txtfile = language_paths[i].upper() + "_SEMEVAL17.txt"
model = fasttext.load_model('subs2vec/subs.' + language_paths[i] + '.bin')

with open(txtfile, "r+") as o:
  words = [line.strip("\n").split("\t") for line in list(o)]
  vecs = []
  s = []
  for w in words:
    w1 = model[w[0]]
    w2 = model[w[1]]
    euclidean = 0
    manhattan = 0
    cosd = 0
    mag1 = 0
    mag2 = 0
    for k in range(len(w1)):
      euclidean += (w1[k] - w2[k]) * (w1[k] - w2[k])
      manhattan += abs(w1[k] - w2[k])
      cosd += w1[k] * w2[k]
      mag1 += w1[k] * w1[k]
      mag2 += w2[k] * w2[k]
    euclidean = euclidean ** 0.5
    cosd = cosd / ((mag1 ** 0.5) * (mag2 ** 0.5))
    if type(cosd) != int:
      cosd = 0
    vecs.append([euclidean, manhattan, cosd])
    s.append(w[2])
  with open(langs[i] + "_wordsim_subsvecs.txt", "w") as tw:
    for k in range(len(vecs)):
      tw.write(str(list(vecs[k])) + "\n")
  with open(langs[i] + "_wordsim_subsscores.txt", "w") as tw:
    for k in range(len(s)):
      tw.write(s[k] + "\n")
