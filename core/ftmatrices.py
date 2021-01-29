'''
For a list of n words, generate an n x n matrix, where row i, column j takes on
value k if word j is the kth nearest neighbor of word i in the given
FastText-compatible model (FastText or subs2vec). Feel free to use your
own word files, models, and out files, or try our defaults. To try our defaults,
download the English FastText bin file from
https://fasttext.cc/docs/en/crawl-vectors.html and place it into a directory
called 'models' within this directory.
'''

# imports
import numpy as np
import string
import fasttext
import argparse

# process command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--word_file",
    help = "name of file with word list, 1 column no headers",
    default = "words/english.txt")
parser.add_argument("--model_file",
    help = "name of FastText-compatible model file to load,"
            + "should be a .bin file",
    default = "models/wiki.en.bin")
parser.add_argument("--out_file",
    help = "name of file to write vectors to",
    default = "english_ft_matrix.txt")
args = parser.parse_args()


# get nearest neighbors of a word given a model, using fasttext.cc API
def get_nn_words(word, model):
  nns = []
  m = list(model.get_nearest_neighbors(word, k = 1000000))
  for s, w in m:
    nns.append(w)
  return nns

mat = []
model = fasttext.load_model(args.model_file)
with open(args.word_file, "r") as words:
  wds = [line.lower().strip("\n") for line in words]
for j in range(len(wds)):
  neighbs = get_nn_words(wds[j], model)
  vals = []
  for n in range(len(wds)):
    try:
      k = neighbs.index(wds[n])
    except:
      k = -1
    vals.append(k)
  mat.append(vals)
with open(args.out_file, "w") as o:
  for q in range(len(wds)):
    o.write(str(mat[q]) + "\n")
