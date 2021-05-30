'''
Run bilingual lexicon induction in a given language to and from English and
get mean of mean cosine similarities as performance metric. Feel free to use
custom input vector files or our defaults. To use our defaults, make sure you
have run either translationftdatagen.py or translationmusedatagen.py first,
depending on which model you'd like to use for your embeddings, so that you
have input vector files that conform to our naming conventions.
'''

# imports
import fasttext
import io
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse

# load a file "sheet" of input vectors
def load(sheet, l):
    x = []
    with open(sheet, "r+") as o:
        for k in range(l):
            l = str(o.readline()).strip("[] \n").split(",")
            lf = [float(i) for i in l]
            x.append(lf)
    x = np.array(x)
    return x

# cosine similarity between prediction vectors and ground-truth vectors
def csim(pred, true):
    total = 0
    for j in range(len(pred)):
        dp, mag1, mag2 = 0, 0, 0
        for k in range(len(pred[j])):
            dp += pred[j][k] * true[j][k]
            mag1 += pred[j][k] * pred[j][k]
            mag2 += true[j][k] * true[j][k]
        if mag1 > 0 and mag2 > 0:
            total += dp / ((mag1 ** 0.5) * (mag2 ** 0.5))
    return total / len(pred)

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--target_train_to_file",
    help = "name of file with target language train vectors for given language to English",
    default = "data/spanish_to_english_spanish_translation_ft_vecs_train.txt")
parser.add_argument("--english_train_to_file",
    help = "name of file with English train vectors for given language to English",
    default = "data/spanish_to_english_spanish_translation_ft_vecs_train.txt")
parser.add_argument("--target_train_from_file",
    help = "name of file with target language train vectors for given language from English",
    default = "data/spanish_from_english_spanish_translation_ft_vecs_train.txt")
parser.add_argument("--english_train_from_file",
    help = "name of file with English train vectors for given language from English",
    default = "data/spanish_from_english_spanish_translation_ft_vecs_train.txt")
parser.add_argument("--target_test_to_file",
    help = "name of file with target language test vectors for given language to English",
    default = "data/spanish_to_english_spanish_translation_ft_vecs_test.txt")
parser.add_argument("--english_test_to_file",
    help = "name of file with English test vectors for given language to English",
    default = "data/spanish_to_english_spanish_translation_ft_vecs_test.txt")
parser.add_argument("--target_test_from_file",
    help = "name of file with target language test vectors for given language from English",
    default = "data/spanish_from_english_spanish_translation_ft_vecs_test.txt")
parser.add_argument("--english_test_from_file",
    help = "name of file with English test vectors for given language from English",
    default = "data/spanish_from_english_spanish_translation_ft_vecs_test.txt")
parser.add_argument("--train_size",
    help = "number of entries in training data",
    default = "5000")
parser.add_argument("--test_size",
    help = "number of entries in testing data",
    default = "1500")
parser.add_argument("--language",
    help = "name of language that the model/words correspond to, "
            + "will be used to name output files",
    default = "spanish")
parser.add_argument("--model_name",
    help = "name of embedding model, "
            + "will be used to name output files",
    default = "ft")
args = parser.parse_args()

# load data files
from_train_x = load(args.english_train_from_file, int(args.train_size))
from_train_y = load(args.target_train_from_file, int(args.train_size))
from_test_x = load(args.english_test_from_file, int(args.test_size))
from_test_y = load(args.target_test_from_file, int(args.test_size))

to_train_x = load(args.target_train_to_file, int(args.train_size))
to_train_y = load(args.english_train_to_file, int(args.train_size))
to_test_x = load(args.target_test_to_file, int(args.test_size))
to_test_y = load(args.english_test_to_file, int(args.test_size))

# train and run model from English
from_model = LinearRegression()
from_model.fit(from_train_x, from_train_y)
from_predictions = from_model.predict(from_test_x)
from_csim = csim(from_predictions, from_test_y)

to_model = LinearRegression()
to_model.fit(to_train_x, to_train_y)
to_predictions = to_model.predict(to_test_x)
to_csim = csim(to_predictions, to_test_y)

with open(args.language + "_" + args.model_name + ".txt", "w") as o:
    o.write("From: " + str(from_csim) + "\n")
    o.write("To: " + str(to_csim) + "\n")


