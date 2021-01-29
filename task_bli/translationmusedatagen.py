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

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
snowball = SnowballStemmer("porter")
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
              'dutch', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']
word_files = [i + '.txt' for i in languages]
vecs = [path_root + i + path_suffix for i in language_paths]
nmax = 50000  # maximum number of word embeddings to load
print("on line 31")
idx = int(sys.argv[1])
loaded = load_vec(vecs[idx], nmax)
englishloaded = load_vec(path_root + "en" + path_suffix, nmax)
print("loaded all vectors")

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

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']

train_to_files = ["TranslationData/" + i + '-en.0-5000.txt' for i in language_paths]
train_from_files = ["TranslationData/en-" + i + '.0-5000.txt' for i in language_paths]
test_to_files = ["TranslationData/" + i + '-en.5000-6500.txt' for i in language_paths]
test_from_files = ["TranslationData/en-" + i + '.5000-6500.txt' for i in language_paths]
i = int(sys.argv[1])

lang = langs[i]
print(lang)

with open(train_to_files[i], "r") as words:
  words = list(words)
  tgt_wds = []
  for w in words:
    tgt_wds.append(w.strip("\n").split()[0])
  eng_wds = []
  for w in words:
    eng_wds.append(w.strip("\n").split()[1])
  tgt_vectors = [get_emb(word, loaded[0], loaded[1], loaded[0], loaded[1], lang) for word in tgt_wds]
  eng_vectors = [get_emb(word, englishloaded[0], englishloaded[1], englishloaded[0], englishloaded[1], "english") for word in eng_wds]
  print(tgt_wds[:6])
  print(eng_wds[:6])
  with open(lang + "to_english_" + lang + "_translationmusevecstrain.txt", "w") as o:
    for v in tgt_vectors:
      o.write(str(list(v)) + "\n")
  with open(lang + "to_english_en_translationmusevecstrain.txt", "w") as o:
    for v in eng_vectors:
      o.write(str(list(v)) + "\n")
print("done with train to")
with open(test_to_files[i], "r") as words:
  words = list(words)
  tgt_wds = [line.strip("\n").split()[0] for line in words]
  eng_wds = [line.strip("\n").split()[1] for line in words]
  tgt_vectors = [get_emb(word, loaded[0], loaded[1], loaded[0], loaded[1], lang) for word in tgt_wds]
  eng_vectors = [get_emb(word, englishloaded[0], englishloaded[1], englishloaded[0], englishloaded[1], "english") for word in eng_wds]
  print(tgt_wds[:6])
  print(eng_wds[:6])
  with open(lang + "to_english_" + lang + "_translationmusevecstest.txt", "w") as o:
    for v in tgt_vectors:
      o.write(str(list(v)) + "\n")
  with open(lang + "to_english_en_translationmusevecstest.txt", "w") as o:
    for v in eng_vectors:
      o.write(str(list(v)) + "\n")
print("done with test to")
with open(train_from_files[i], "r") as words:
  words = list(words)
  tgt_wds = [line.strip("\n").split()[1] for line in words]
  eng_wds = [line.strip("\n").split()[0] for line in words]
  print(tgt_wds[:6])
  print(eng_wds[:6])
  tgt_vectors = [get_emb(word, loaded[0], loaded[1], loaded[0], loaded[1], lang) for word in tgt_wds]
  eng_vectors = [get_emb(word, englishloaded[0], englishloaded[1], englishloaded[0], englishloaded[1], "english") for word in eng_wds]
  with open(lang + "from_english_" + lang + "_translationmusevecstrain.txt", "w") as o:
    for v in tgt_vectors:
      o.write(str(list(v)) + "\n")
  with open(lang + "from_english_en_translationmusevecstrain.txt", "w") as o:
    for v in eng_vectors:
      o.write(str(list(v)) + "\n")
print("done with train from")
with open(test_from_files[i], "r") as words:
  words = list(words)
  tgt_wds = [line.strip("\n").split()[1] for line in words]
  eng_wds = [line.strip("\n").split()[0] for line in words]
  print(tgt_wds[:6])
  print(eng_wds[:6])
  tgt_vectors = [get_emb(word, loaded[0], loaded[1], loaded[0], loaded[1], lang) for word in tgt_wds]
  eng_vectors = [get_emb(word, englishloaded[0], englishloaded[1], englishloaded[0], englishloaded[1], "english") for word in eng_wds]
  with open(lang + "from_english_" + lang + "_translationmusevecstest.txt", "w") as o:
    for v in tgt_vectors:
      o.write(str(list(v)) + "\n")
  with open(lang + "from_english_en_translationmusevecstest.txt", "w") as o:
    for v in eng_vectors:
      o.write(str(list(v)) + "\n")
print("done with test from")
