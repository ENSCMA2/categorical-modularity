'''
Generate vectorized training and testing data in any FastText-compatible model
for bilingual lexicon induction in a given language to and from English.
Feel free to use our defaults or input your own word files and models. Note that
our default file format is .txt. To use our defaults, download the English and
Spanish FastText bin files from https://fasttext.cc/docs/en/crawl-vectors.html
and place them into a directory called 'models' within this directory.
Outputs 8 files: 4 training & 4 testing, 4 to & 4 from, 4 English & 4
non-English. Output files are named
[target lang]_[to/from]_english_[english/target lang]_translation_[model name]_vecs_[train/test].txt
'''

# imports
import fasttext
import sys
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

train_to_file = args.train_to_file
train_from_file = args.train_from_file
test_to_files = args.test_to_file
test_from_files = args.test_from_file

lang = args.language
model_name = args.model_name

model = fasttext.load_model(args.target_model_file)
englishmodel = fasttext.load_model(args.english_model_file)

with open(train_to_file, "r") as words:
    words = list(words)
    tgt_wds = [line.strip("\n").split()[0] for line in words]
    eng_wds = [line.strip("\n").split()[1] for line in words]
    tgt_vectors = [model[tgt_wds[i]] for i in range(len(tgt_wds))]
    eng_vectors = [englishmodel[eng_wds[i]] for i in range(len(eng_wds))]
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
    tgt_vectors = [model[tgt_wds[i]] for i in range(len(tgt_wds))]
    eng_vectors = [englishmodel[eng_wds[i]] for i in range(len(eng_wds))]
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
    tgt_vectors = [model[tgt_wds[i]] for i in range(len(tgt_wds))]
    eng_vectors = [englishmodel[eng_wds[i]] for i in range(len(eng_wds))]
    with open("data/" + lang + "_from_english_" + lang + "_translation_"
               + model_name + "_vecs_train.txt", "w") as o:
        for v in tgt_vectors:
            o.write(str(list(v)) + "\n")
    with open("data/" + lang + "_from_english_english_translation_" + model_name
               + "_vecs_train.txt", "w") as o:
        for v in eng_vectors:
            o.write(str(list(v)) + "\n")

with open(test_from_file, "r") as words:
    words = list(words)
    tgt_wds = [line.strip("\n").split()[1] for line in words]
    eng_wds = [line.strip("\n").split()[0] for line in words]
    tgt_vectors = [model[tgt_wds[i]] for i in range(len(tgt_wds))]
    eng_vectors = [englishmodel[eng_wds[i]] for i in range(len(eng_wds))]
    with open("data/" + lang + "_from_english_" + lang + "_translation_"
               + model_name + "_vecs_test.txt", "w") as o:
        for v in tgt_vectors:
            o.write(str(list(v)) + "\n")
    with open("data/" + lang + "_from_english_english_translation_" + model_name
               + "_vecs_test.txt", "w") as o:
        for v in eng_vectors:
            o.write(str(list(v)) + "\n")

