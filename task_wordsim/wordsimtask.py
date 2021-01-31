'''
Run the word similarity task on a given language/model. Input is 3-dimensional
vectors consisting of Euclidean distance, Manhattan distance, and cosine
distance. Output is mean squared error loss averaged over num_trials trials.
Output is printed to console.
'''

# imports
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_file",
    help = "name of file with input data",
    default = "data/english_wordsim_ft_vecs.txt")
parser.add_argument("--label_file",
    help = "name of file with similarity scores",
    default = "data/english_wordsim_ft_scores.txt")
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

datafile = args.data_file
labelfile = args.label_file

dataset_size = int(args.dataset_size)
numtrials = int(args.num_trials)
possibilities = [i for i in range(dataset_size)]
loss_avg = 0

# running trials
for i in range(numtrials):
    test = random.sample(possibilities,
                         int(dataset_size * (1 - float(args.train_proportion))))
    traindata = []
    testdata = []
    trainlabels = []
    testlabels = []
    with open(datafile, "r+") as o:
        for k in range(dataset_size):
            l = [float(y) for y in str(o.readline()).strip("[] \n").split(",")]
            if k in test:
                testdata.append(l)
            else:
                traindata.append(l)
    with open(labelfile, "r+") as o:
        for k in range(dataset_size):
            s = float(str(o.readline()))
            if k in test:
                testlabels.append(s)
            else:
                trainlabels.append(s)
    traindata = np.array(traindata)
    testdata = np.array(testdata)
    trainlabels = np.array(trainlabels)
    testlabels = np.array(testlabels)
    model = LinearRegression()
    model.fit(traindata, trainlabels)
    preds = model.predict(testdata)
    loss = 0
    for num in range(len(preds)):
        loss += (preds[num] - testlabels[num]) * (preds[num] - testlabels[num])
    loss = loss / len(preds)
    loss_avg += loss / numtrials

print(loss_avg)






