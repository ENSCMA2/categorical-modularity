'''
Calculate Spearman correlation of a given general modularity metric with a given
downstream task performance metric. The example below shows Level 3, k = 2
modularities against wordsim task results. Results printed to console.
'''

# imports
import csv
from scipy.stats import spearmanr as sr

# processing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modularity_file",
    help = "name of file with modularity scores, 1-column .csv",
    default = "data/3_2.csv")
parser.add_argument("--downstream_file",
    help = "name of file with downstream performance metrics, 1-column .csv",
    default = "data/loss.csv")
args = parser.parse_args()

mods = []
with open(args.modularity_file, newline = "") as csvfile:
    reader = csv.reader(csvfile)
    mods = [float(row[0]) for row in reader]

metrics = []
with open(args.downstream_file, newline = "") as csvfile:
    reader = csv.reader(csvfile)
    metrics = [float(row[0]) for row in reader]

print(sr(mods, metrics))
