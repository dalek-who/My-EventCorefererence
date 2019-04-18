#%%
import networkx as nx
from pandas import read_csv, DataFrame, Series
from itertools import product
#%%
path = "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_predict/train-large_wdoc_wtpc-lr_5e-6-batch_65-epoch_5--test-large_wdoc_wtpc/BERT_predict.tsv"
df = read_csv(path, sep="\t")
#%%
df_positive = df.loc[df.loc[:, "# pred label"]==1, :]


#%%
nodes = set(df.loc[:,"#1 ID"].to_list()+df.loc[:,"#2 ID"].to_list())
links = df.loc[df.loc[:, "# pred label"]==1, ["#1 ID", "#2 ID"]].to_dict(orient='split')['data']
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(links)

clusters = [list(c) for c in nx.connected_components(G)]

df_clusters_links = DataFrame(columns=["#1 ID", "#2 ID", "# 0 likelihood", "#1 likelihood", "# pred label"])
for c in clusters:
    for u, v in product(c, repeat=2):
        sr = Series({"#1 ID": u, "#2 ID": v, "# pred label": 1})
        df_clusters_links = df_clusters_links.append(sr, ignore_index=True)

#%%
new_links = []
for c in clusters:
    for u, v in product(c, repeat=2):
        new_links.append(f"{u}++{v}")

df_new: DataFrame = df.copy()
df_new.loc[(df_new["#1 ID"]+"++"+df_new["#2 ID"]).isin(new_links), "# pred label"] = 1
