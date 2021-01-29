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

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

path_root = '/Users/karinahalevy/Code/MUSE/data/wiki.multi.'
path_suffix = '.vec'
languages = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']

def average(embeddings):
    num_embeddings = len(embeddings)
    new_embedding = []
    for i in range(len(embeddings[0])):
        rs = 0
        for j in range(num_embeddings):
            rs += embeddings[j][i]
        new_embedding.append(rs / num_embeddings)
    return np.array(new_embedding)

def get_emb(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang, K=30):
    word2id = {v: k for k, v in src_id2word.items()}
    tok = word.split()
    embs = []
    for i in tok:
        try:
            e = src_emb[word2id[i]]
        except:
            try:
                e = src_emb[word2id[stemmer.stem(i)]]
            except:
                try:
                    e = src_emb[word2id[lemmatizer.lemmatize(i)]]
                except:
                    try:
                        e = src_emb[word2id[snowball.stem(i)]]
                    except:
                        try:
                            e = src_emb[word2id[SnowballStemmer(lang).stem(i)]]
                        except:
                            e = []
        if len(list(e)) > 0:
          embs.append(e)
    if len(embs) == 0:
        word_emb = np.array([0] * 300)
    else:
        word_emb = average(embs)
    return word_emb

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang, K=30):
    word2id = {v: k for k, v in src_id2word.items()}
    tok = word.split()
    embs = []
    for i in tok:
        try:
            e = src_emb[word2id[i]]
        except:
            try:
                e = src_emb[word2id[stemmer.stem(i)]]
            except:
                try:
                    e = src_emb[word2id[lemmatizer.lemmatize(i)]]
                except:
                    try:
                        e = src_emb[word2id[snowball.stem(i)]]
                    except:
                        try:
                            e = src_emb[word2id[SnowballStemmer(lang).stem(i)]]
                        except:
                            e = []
        if len(list(e)) > 0:
          embs.append(e)
    if len(embs) == 0:
        word_emb = np.array([0] * 300)
    else:
        word_emb = average(embs)
    # print("Nearest neighbors of \"%s\":" % word)
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    nns = []
    for i, idx in enumerate(k_best):
        nns.append((scores[idx], tgt_id2word[idx]))
    return nns

i = int(sys.argv[1])
print(languages[i])
txtfile = language_paths[i].upper() + "_SEMEVAL17.txt"

vecs = path_root + language_paths[i] + path_suffix
nmax = 50000  # maximum number of word embeddings to load
loaded = load_vec(vecs, nmax)

with open(txtfile, "r+") as o:
  words = [line.strip("\n").split("\t") for line in list(o)]
  vecs = []
  s = []
  for w in words:
    w1 = get_emb(w[0], loaded[0], loaded[1],
      loaded[0], loaded[1], languages[i])
    w2 = get_emb(w[1], loaded[0], loaded[1],
      loaded[0], loaded[1], languages[i])
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
  with open(languages[i] + "_wordsim_musevecs.txt", "w") as tw:
    for k in range(len(vecs)):
      tw.write(str(list(vecs[k])) + "\n")
  with open(languages[i] + "_wordsim_musescores.txt", "w") as tw:
    for k in range(len(s)):
      tw.write(s[k] + "\n")
