'''
Run the sentiment analysis task on one language/model. To use our defaults,
make sure you have run either ftmoviegen.py or musemoviegen.py first so that
you have vectorized data to feed into this model. Assumes that data is 5k
positive and 5k negative (doesn't take in a label file). Output (mean accuracy
and precision over num_trials trials) is placed in the 'data' directory and
named [language]_movies_[model name]_metrics_[num trials].txt
'''

# imports
import csv
import numpy as np
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import random
import argparse

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_file",
    help = "name of file with input data",
    default = "data/english_ft_movievecs.txt")
parser.add_argument("--model_name",
    help = "name of model, used to name output file",
    default = "ft")
parser.add_argument("--num_trials",
    help = "number of trials to run for",
    default = "30")
parser.add_argument("--dataset_size",
    help = "number of entries in dataset",
    default = "500")
parser.add_argument("--train_proportion",
    help = "proportion of dataset to be used for training, decimal format, 0 to 1",
    default = "0.8")
parser.add_argument("--language",
    help = "name of language your words/model correspond to",
    default = "english")
args = parser.parse_args()

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian']

# load data with a given train-test split
def load_data(sheet, indices, size):
    x = []
    with open((sheet), "r+") as o:
        for k in range(size):
            l = str(o.readline()).strip("[] \n").split(",")
            lf = [float(i) for i in l]
            x.append(lf)
    x = np.array(x)
    y = np.array(['positive'] * int(size / 2) + ['negative'] * int(size / 2))
    X_TRAIN = np.array([x[j] for j in range(len(x)) if j not in indices])
    Y_TRAIN = np.array([y[j] for j in range(len(y)) if j not in indices])
    X_TEST = np.array([x[j] for j in range(len(x)) if j in indices])
    Y_TEST = np.array([y[j] for j in range(len(y)) if j in indices])
    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

def average(numbers):
    return sum(numbers)/len(numbers)

numtrials = args.num_trials

storage = {"accuracy": [], "precision": []}

for i in range(numtrials):
    indices = random.sample([num for num in range(0, int(args.dataset_size))],
                      int(float(args.train_proportion) * int(args.dataset_size)))
    train_data, test_data, train_labels, test_labels = load_data(args.data_file,
                                                                 indices,
                                                                 int(args.dataset_size))
    classifier = LinearSVC()
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, pos_label = 'positive')
    storage["accuracy"].append(accuracy)
    storage["precision"].append(precision)

with open("data/" + args.language + "_movies_" + args.model_name + "_metrics"
          + args.numtrials + ".txt", "w") as o:
    o.write("Accuracy: " + str(average(storage["accuracy"])) + "\n")
    o.write("Precision: " + str(average(storage["precision"])))
