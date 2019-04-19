#%%
import csv
from pandas import read_csv
#%%
path_MRPC = "/users/wangyuanzheng/event_coreference/my-ev-coref/datasets/MRPC/dev.tsv"
tsv_MRPC = read_csv(path_MRPC, sep="\t", quoting=csv.QUOTE_NONE)
#%%
path_ECB = '/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_input/test-large_wdoc_wtpc/dev.tsv'
tsv_ECB = read_csv(path_ECB, sep="\t")
