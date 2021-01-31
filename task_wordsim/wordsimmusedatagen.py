'''
Generate data for word similarity task. Input is a word pair file conforming to
SEMEVAL17 txt file format and a MUSE-compatible language model bin file.
Output is 2 files, 1 with regression inputs [Euclidean, Manhattan, cosine] and
1 with similarity scores directly from SEMEVAL. To use our defaults, download
the English and MUSE model from the table at
https://github.com/facebookresearch/MUSE#download and place it into a
directory called 'models' within this directory, and make sure you have run
wordsimtrans.py first so that you have the full translated datsets. Output file
of regression inputs is named [language name]_wordsim_[model name]_vecs.txt, and
output file of scores is named [language name]_wordsim_[model name]_scores.txt,
both placed into 'data' directory.
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
parser.add_argument("--word_file",
    help = "name of file with word pairs and similarities, 3-column .txt",
    default = "data/EN_SEMEVAL17.txt")
parser.add_argument("--model_file",
    help = "name of MUSE-compatible model file to load",
    default = "models/wiki.multi.en.bin")
parser.add_argument("--model_name",
    help = "name of MUSE-compatible model, used to name output file",
    default = "ft")
parser.add_argument("--language",
    help = "name of language your words/model correspond to",
    default = "english")
args = parser.parse_args()

# initialize some preprocessors and constants
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
nmax = 200000  # maximum number of word embeddings to load
txtfile = args.word_file

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

with open(txtfile, "r+") as o:
    words = [line.strip("\n").split("\t") for line in list(o)]
    vecs = []
    s = []
    for w in words:
        w1 = get_emb(w[0], loaded[0], loaded[1],
                     loaded[0], loaded[1], args.language)
        w2 = get_emb(w[1], loaded[0], loaded[1],
                     loaded[0], loaded[1], args.language)
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
    with open("data/" + args.language + "_wordsim_" + args.model_name
              + "_vecs.txt", "w") as tw:
        for k in range(len(vecs)):
            tw.write(str(list(vecs[k])) + "\n")
    with open("data/" + args.language + "_wordsim_" + args.model_name
              + "_scores.txt", "w") as tw:
        for k in range(len(s)):
            tw.write(s[k] + "\n")
