'''
Calculate general categorical modularity. To generate a matrix, run ftmatrices.py
or musematrices.py depending on what model you want. Then, paste the output file
contents into an Excel sheet, split the data to columns, get rid of the []
characters, and re-download as a csv. Results are printed to console.
'''

# imports
import numpy as np
import csv 
from sklearn.neighbors import kneighbors_graph
import argparse

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--categories_file",
    help = "name of file with category labels, 1-column .csv, no headers",
    default = "words/categories_2.csv")
parser.add_argument("--matrix_file",
    help = "name of csvfile with square k-NN matrix",
    default = "data/muse_finnish.csv")
args = parser.parse_args()

categories = []
with open(args.categories_file) as csvfile:    
	csvReader = csv.reader(csvfile)    
	for row in csvReader:        
		categories.append(row[0])

# get csv of the category names
counters = []
i = 0
count = "Living Things"  # first elt in categories file
for elt in categories:
    if elt == count:
        i += 1
    else:
        counters.append(i)
        i = 1
        count = elt
counters.append(i)  # for the last category

file = open(args.matrix_file)  # read in your matrix, no word labels
M = np.loadtxt(file, delimiter = ",")

lst = [2, 3, 4]
for k in lst:
    knn = kneighbors_graph(M, k, mode = 'connectivity', include_self = True) 
    knnmatrix = knn.toarray()

    # counting a_c (modularity paper notation, changing l of languages for c of categories)
    ac = []
    t = 0
    for c in counters:  # for each category
        cjtotal = 0
        for i in range(t, t + c):  # how many words in that category
            for j in knnmatrix[i]:  # count degree of that word / node
                if (j == 1):
                    cjtotal += 1
        ac.append(cjtotal)
        t += c

    # we need to divide by 2m according to the formula
    m = 0 
    for i in range(len(knnmatrix[0])):
        for j in range(len(knnmatrix[0])):
            if knnmatrix[i, j] == 1:
                m += 1

  # now the true ac
    for i in range(len(ac)):
        ac[i] = ac[i] / m

    # now we compute ell (modularity paper), called ecc here (fraction of edges within the same category)
    ecc = []
    t = 0
    for c in counters:
        ecctotal = 0
        for i in range(t, t + c):
            for j in range(t, t + c):
                if (knnmatrix[i, j] == 1):
                    ecctotal += 1
        ecc.append(ecctotal)
        t += c

    # actually divided by 2m
    for i in range(len(ecc)):
        ecc[i] = ecc[i] / m

    # Given C total categories, we calculate the overall modularity Q
    Q = 0
    for i in range(len(counters)):
        Q += ecc[i] - ac[i] * ac[i]

    # finally, we normalize
    Qmax = 0
    for i in range(len(counters)):
        Qmax += ac[i] * ac[i]
    Qmax = 1 - Qmax
    Qnorm = Q/Qmax

    print(Qnorm)
