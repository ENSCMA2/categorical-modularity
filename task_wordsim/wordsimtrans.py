'''
Generate translated version of SemEval 2017 English word pair list. Feel free
to use our defaults or try your own languages and files. Output file is named
[2-letter target language code in upper case]_SEMEVAL17.txt and is placed in
the 'data' directory.
'''

# imports
import argparse
from google_trans_new import google_translator

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--word_file",
    help = "name of file with word pairs and similarities, 3-column .txt",
    default = "data/EN_SEMEVAL17.txt")
parser.add_argument("--source_language",
    help = "2-letter code of language you want to translate from",
    default = "en")
parser.add_argument("--target_language",
    help = "2-letter code of language you want to translate to",
    default = "ru")
args = parser.parse_args()

# initializations
translator = google_translator()
txtfile = args.word_file

with open(txtfile, "r+") as o:
    words = [line.strip("\n").split("\t") for line in list(o)]
    p1 = []
    p2 = []
    s = []
    for w in words:
        w1 = translator.translate(w[0], lang_src = args.source_language,
          lang_tgt = args.target_language)
        w2 = translator.translate(w[1], lang_src = args.source_language,
          lang_tgt = args.target_language)
        p1.append(w1)
        p2.append(w2)
        # useful to print as you go in case you hit the rate limit
            # and want to save your progress
        print(w1 + "\t" + w2 + "\t" + w[2])
        s.append(w[2])
    with open("data/" + args.language.upper() + "_SEMEVAL17.txt", "w") as tw:
        for k in range(len(p1)):
            tw.write(p1[k] + "\t" + p2[k] + "\t" + s[k] + "\n")
