import csv
from pandas import read_csv

tsv = read_csv("./dev.tsv", sep="\t", quoting=csv.QUOTE_NONE)
print(tsv.head())
