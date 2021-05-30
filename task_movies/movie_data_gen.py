'''
Generate translated IMDB movie data. Input is a 2-column tsv with reviews in
first column and ratings in second column. Output is a txt with just reviews,
placed into the 'data' directory and named IMDB_[target language].txt
'''

# imports
import csv
import numpy as np
from google_trans_new import google_translator
import random
import argparse

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_file",
    help = "name of movie data file, 2-column tsv, 1st column is review, 2nd column is 0/1 label, no header",
    default = "data/IMDBselected.tsv")
parser.add_argument("--target_language_name",
    help = "name of language you want to translate to",
    default = "spanish")
parser.add_argument("--target_language_code",
    help = "2-letter code of language you want to translate to",
    default = "es")
args = parser.parse_args()

# initializing translator
translator = google_translator()

# open English file, translate line by line, write to target file
with open(args.data_file, encoding = "ISO-8859-1", mode='r+') as o:
    lines = csv.reader(o, delimiter = '\t')
    data = list(lines)
    x = [point[0] for point in data]
    with open("data/IMDB_" + args.target_language_name + ".txt", "w") as n:
        for point in x:
          t = translator.translate(point, lang_src = args.source_language,
                                   lang_tgt = args.target_language_code)
          if type(t) != str:
              t = t[0]
          n.write(t + "\n")
