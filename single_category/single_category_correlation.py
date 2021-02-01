'''
Calculate single-category modularity correlations with a given performance
metric. You can enter your own files or try our defaults! Note that
we recommend using csv files, as this script reads data using csv.
'''

# imports
import csv
import argparse
from scipy.stats import spearmanr as sr

# process command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modularity_file",
    help = "name of file with modularity scores, columns = categories,"
            + "1st row = column headers, no row headers",
    default = "data/3_2.csv")
parser.add_argument("--metrics_file",
    help = "name of file with task performance metrics, 1 column, no headers,"
            + "assuming length aligns with modularity_file",
    default = "movies_accuracy.csv")
parser.add_argument("--out_file",
    help = "name of file to write correlation to, if desired",
    default = "out.txt")
args = parser.parse_args()

# scores: stores modularity scores - outer key = k, inner key = level
scores = {}

# reading in modularity data
with open(args.modularity_file, newline = "") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for i in range(len(rows[0])):
        scores[rows[0][i]] = [float(score[i]) for score in rows[1:]]

metric = []

# reading in performance data, assuming 1-column list w/ no header
with open(args.metrics_file, "r") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for i in range(len(rows)):
        metric.append(float(rows[i][0]))

# writes results to a file, separates columns by commas
with open(args.out_file, "w") as o:
    o.write("category,correlation\n") # column headers
    for cat in scores.keys():
        o.write(cat + "," + str(sr(scores[cat], metric).correlation) + "\n")
