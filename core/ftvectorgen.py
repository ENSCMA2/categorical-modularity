'''
Generate FastText-compatible embeddings of a list of words. Can use custom input
and output files and models or try our defaults. Works for FastText or
subs2vec models and any other model that can be loaded with fasttext. To use
our defaults, download the English FastText bin file from
https://fasttext.cc/docs/en/crawl-vectors.html and place it into a directory
called 'models' within this directory.
'''

# imports
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
    default = "english_ft_vecs.txt")
args = parser.parse_args()

# load the embedding model
model = fasttext.load_model(args.model_file)

# open words and convert them to embeddings
with open(args.word_file, "r") as words:
    wds = [line.lower().strip("\n") for line in words]
    vectors = [model[wds[i]] for i in range(len(wds))]

# write embeddings to output file
with open(out_file, "w") as o:
    for v in vectors:
        o.write(str(list(v)) + "\n")
