'''
For a list of n words, generate an n x n matrix, where row i, column j takes on
value k if word j is the kth nearest neighbor of word i in the given
MUSE-compatible model. Feel free to use your own word files, models, and out
files, or try our defaults. To use our defaults, download the English MUSE model
from the table at https://github.com/facebookresearch/MUSE#download and place it
into a directory called 'models' within this directory.
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

# process command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--word_file",
    help = "name of file with word list, 1 column, no headers,"
            + "1st row = column headers, no row headers",
    default = "words/english.txt")
parser.add_argument("--model_file",
    help = "name of MUSE-compatible model file to load,"
            + "should be a .vec file",
    default = "models/wiki.multi.en.vec")
parser.add_argument("--out_file",
    help = "name of file to write vectors to",
    default = "english_muse_matrix.csv")
parser.add_argument("--language",
    help = "language of model, needed for word stemming",
    default = "english")
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

# load the MUSE model
loaded = load_vec(args.model_file, nmax)

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

# create the matrix and write it to a file
mat = []
with open(args.word_file, "r") as words:
    wds = [line.lower().strip("\n").strip(" ") for line in words]
for j in range(len(wds)):
    vals = get_nn_word(wds[j], wds, loaded[0], loaded[1], loaded[0], loaded[1],
                       args.language)
    mat.append(vals)
with open(args.out_file, "w") as o:
    for q in range(len(wds)):
        o.write(str(mat[q]).strip("[]") + "\n")


