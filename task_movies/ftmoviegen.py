'''
Generate vectorized data from IMDB raw text reviews using FastText-compatible
models. To use our defaults, download the English FastText bin file from
https://fasttext.cc/docs/en/crawl-vectors.html and place it into a directory
called 'models' within this directory, and make sure you have run
moviedatagen.py first so you have the full translated datasets.
'''

# imports
import random
import fasttext
import argparse

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_file",
    help = "name of file with word pairs and similarities, 3-column .txt",
    default = "data/IMDB_english.txt")
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

model = fasttext.load_model(args.model_file)

with open(args.data_file, "r+") as o:
    text = [line.lower().strip("\n") for line in o]
    with open("data/" + args.language + "_" + args.model_name + "_movievecs.txt",
              "w") as n:
        for j in range(len(text)):
            vec = list(model[text[j]])
            n.write(str(vec) + "\n")


