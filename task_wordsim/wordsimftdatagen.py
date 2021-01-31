'''
Generate data for word similarity task. Input is a word pair file conforming to
SEMEVAL17 txt file format and a FastText-compatible language model bin file.
Output is 2 files, 1 with regression inputs [Euclidean, Manhattan, cosine] and
1 with similarity scores directly from SEMEVAL. To use our defaults, download
the English FastText bin file from
https://fasttext.cc/docs/en/crawl-vectors.html and place it into a directory
called 'models' within this directory, and make sure you have run wordsimtrans.py
first so you have the full translated datasets. Output file of regression inputs
is named [language name]_wordsim_[model name]_vecs.txt, and output file of
scores is named [language name]_wordsim_[model name]_scores.txt,
both placed into 'data' directory.
'''

# imports
import fasttext
import argparse
import numpy as np

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--word_file",
    help = "name of file with word pairs and similarities, 3-column .txt",
    default = "data/EN_SEMEVAL17.txt")
parser.add_argument("--model_file",
    help = "name of FastText-compatible model file to load",
    default = "models/wiki.en.bin")
parser.add_argument("--model_name",
    help = "name of FastText-compatible model, used to name output file",
    default = "ft")
parser.add_argument("--language",
    help = "name of language your words/model correspond to",
    default = "english")
args = parser.parse_args()

txtfile = args.word_file
model = fasttext.load_model(args.model_file)

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
