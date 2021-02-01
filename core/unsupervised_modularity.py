'''
Compute the modularity of an unsupervised network generated using a community
detection algorithm. Results printed to console.
'''

# imports
import networkx as nx
import numpy as np
import csv 
import networkx.algorithms.community as nx_comm
from sklearn.neighbors import kneighbors_graph
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics.cluster import normalized_mutual_info_score
import argparse

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--matrix_file",
    help = "name of csv file with square k-NN matrix",
    default = "data/muse_finnish.csv")
args = parser.parse_args()

file = open(args.matrix_file)
M = np.loadtxt(file, delimiter=",")

values = [2, 3, 4]
for k in values:
    knn = kneighbors_graph(M, k, mode='connectivity', include_self=True) 
    knnmatrix = knn.toarray()
    G = nx.from_numpy_matrix(np.array(knnmatrix))

    c = list(greedy_modularity_communities(G))

    categories = []
    emerging_labels = []
    for i in range(500):
        emerging_labels.append(0)

    for i in range(len(c)):
        cluster = []
        for x in c[i]:
            cluster.append(x)
            emerging_labels[x] = i 
        cluster.sort()
        categories.append(cluster)

    result = nx_comm.modularity(G, categories)
    print("k = " + str(k) + ": " + str(result))

