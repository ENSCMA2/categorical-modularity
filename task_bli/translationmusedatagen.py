'''
Generate vectorized training and testing data in any MUSE-compatible model
for bilingual lexicon induction in a given language to and from English.
Feel free to use our defaults or input your own word files and models. Note that
our default file format is .txt. To use our defaults, download the English and
Spanish MUSE models from the table at
https://github.com/facebookresearch/MUSE#download and place them into a
directory called 'models' within this directory. Outputs 8 files: 4 training & 4
testing, 4 to & 4 from, 4 English & 4 non-English. Output files are named
[target lang]_[to/from]_english_[english/target lang]_translation_[model name]_vecs_[train/test].txt
and are placed into the 'data' directory.
'''

# imports
import io
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import argparse

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_to_file",
    help = "name of file with train word list for given language to English,"
    + " 2-column space-separated txt file, 1st column is non-English,"
    + " no headers",
    default = "data/es-en.0-5000.txt")
parser.add_argument("--train_from_file",
    help = "name of file with train word list for given language from English,"
    + " 2-column space-separated txt file, 2nd column is non-English,"
    + " no headers",
    default = "data/en-es.0-5000.txt")
parser.add_argument("--test_to_file",
    help = "name of file with test word list for given language to English,"
    + " 2-column space-separated txt file, 1st column is non-English,"
    + " no headers",
    default = "data/es-en.5000-6500.txt")
parser.add_argument("--test_from_file",
    help = "name of file with test word list for given language from English,"
    + " 2-column space-separated txt file, 2nd column is non-English,"
    + " no headers",
    default = "data/en-es.5000-6500.txt")
parser.add_argument("--target_model_file",
    help = "name of FastText-compatible model file to load for non-English"
            + " language, should be a .bin file",
    default = "models/wiki.es.bin")
parser.add_argument("--english_model_file",
    help = "name of FastText-compatible model file to load for English"
            + " language, should be a .bin file",
    default = "models/wiki.en.bin")
parser.add_argument("--language",
    help = "name of language that the model/words correspond to, "
    + "will be used to name output files",
    default = "spanish")
parser.add_argument("--model_name",
    help = "name of embedding model, "
    + "will be used to name output files",
    default = "ft")
args = parser.parse_args()


# initialize some preprocessors and constants
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
nmax = 200000  # maximum number of word embeddings to load

# loads embedding model from model_file path
# source: https://github.com/facebookresearch/MUSE/blob/master/demo.ipynb
def load_vec(emb_path, nmax = 200000):
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

# load the MUSE models
loaded = load_vec(args.target_model_file, nmax)
englishloaded = load_vec(args.english_model_file, nmax)

# calculate sentence embedding by taking mean of component words
def average(embeddings):
    num_embeddings = len(embeddings)
    new_embedding = []
    for i in range(len(embeddings[0])):
        rs = 0
        for j in range(num_embeddings):
            rs += embeddings[j][i]
        new_embedding.append(rs / num_embeddings)
    return np.array(new_embedding)

# get embedding of a word given a source and target space and a language
# source: code modified from
    # https://github.com/facebookresearch/MUSE/blob/master/demo.ipynb
def get_emb(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang):
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

# get nearest neighbors of a word
def get_nns(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang):
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
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-1 * len(scores):][::-1]
    nns = []
    for i, idx in enumerate(k_best):
        nns.append((scores[idx], tgt_id2word[idx]))
    nn_w = []
    for s, w in nns:
        nn_w.append(w)

# get nearest neighbors for all 1 word across all words
def get_nn_word(word, words, src_emb, src_id2word, tgt_emb, tgt_id2word, lang):
  tr = []
  n = get_nns(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang)
  for w in words:
    try:
      tr.append(n.index(w))
    except:
      tr.append(-1)
  return tr

train_to_file = args.train_to_file
train_from_file = args.train_from_file
test_to_files = args.test_to_file
test_from_files = args.test_from_file

lang = args.language
model_name = args.model_name

with open(train_to_file, "r") as words:
    words = list(words)
    tgt_wds = [line.strip("\n").split()[0] for line in words]
    eng_wds = [line.strip("\n").split()[1] for line in words]
    tgt_vectors = [get_emb(word, loaded[0], loaded[1], loaded[0], loaded[1], lang)
                   for word in tgt_wds]
    eng_vectors = [get_emb(word, englishloaded[0], englishloaded[1],
                           englishloaded[0], englishloaded[1], "english")
                   for word in eng_wds]
    with open("data/" + lang + "_to_english_" + lang + "_translation_"
              + model_name + "_vecs_train.txt", "w") as o:
        for v in tgt_vectors:
            o.write(str(list(v)) + "\n")
    with open("data/" + lang + "_to_english_english_translation_" + model_name
              + "_vecs_train.txt", "w") as o:
        for v in eng_vectors:
            o.write(str(list(v)) + "\n")

with open(test_to_file, "r") as words:
    words = list(words)
    tgt_wds = [line.strip("\n").split()[0] for line in words]
    eng_wds = [line.strip("\n").split()[1] for line in words]
    tgt_vectors = [get_emb(word, loaded[0], loaded[1],
                           loaded[0], loaded[1], lang)
                   for word in tgt_wds]
    eng_vectors = [get_emb(word, englishloaded[0], englishloaded[1],
                           englishloaded[0], englishloaded[1], "english")
                   for word in eng_wds]
    with open("data/" + lang + "_to_english_" + lang + "_translation_"
              + model_name + "_vecs_test.txt", "w") as o:
        for v in tgt_vectors:
            o.write(str(list(v)) + "\n")
    with open("data/" + lang + "_to_english_english_translation_" + model_name
              + "_vecs_test.txt", "w") as o:
        for v in eng_vectors:
            o.write(str(list(v)) + "\n")

with open(train_from_file, "r") as words:
  words = list(words)
  tgt_wds = [line.strip("\n").split()[1] for line in words]
  eng_wds = [line.strip("\n").split()[0] for line in words]
  tgt_vectors = [get_emb(word, loaded[0], loaded[1], loaded[0], loaded[1], lang)
                 for word in tgt_wds]
  eng_vectors = [get_emb(word, englishloaded[0], englishloaded[1], englishloaded[0],
                         englishloaded[1], "english")
                 for word in eng_wds]
  with open("data/" + lang + "_from_english_" + lang + "_translation_"
            + model_name + "_vecs_train.txt", "w") as o:
      for v in tgt_vectors:
          o.write(str(list(v)) + "\n")
  with open("data/" + lang + "_from_english_english_translation_" + model_name
            + "_vecs_train.txt", "w") as o:
      for v in eng_vectors:
          o.write(str(list(v)) + "\n")

with open(test_from_files[i], "r") as words:
  words = list(words)
  tgt_wds = [line.strip("\n").split()[1] for line in words]
  eng_wds = [line.strip("\n").split()[0] for line in words]
  tgt_vectors = [get_emb(word, loaded[0], loaded[1], loaded[0], loaded[1], lang)
                 for word in tgt_wds]
  eng_vectors = [get_emb(word, englishloaded[0], englishloaded[1],
                         englishloaded[0], englishloaded[1], "english")
                 for word in eng_wds]
  with open("data/" + lang + "_from_english_" + lang + "_translation_"
            + model_name + "_vecs_test.txt", "w") as o:
      for v in tgt_vectors:
          o.write(str(list(v)) + "\n")
  with open("data/" + lang + "_from_english_english_translation_" + model_name
            + "_vecs_test.txt", "w") as o:
      for v in eng_vectors:
          o.write(str(list(v)) + "\n")
