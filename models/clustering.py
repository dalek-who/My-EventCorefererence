import networkx as nx
from pandas import read_csv, DataFrame, Series
from itertools import product
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

#%%
input_path = "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_predict/train-large_wdoc_wtpc-lr_5e-6-batch_65-epoch_5--test-large_wdoc_wtpc/BERT_predict.tsv"
input_df = read_csv(input_path, sep="\t")


def connected_components_clustering(df):
    nodes = set(df.loc[:, "#1 ID"].to_list() + df.loc[:, "#2 ID"].to_list())
    links = df.loc[df.loc[:, "# pred label"] == 1, ["#1 ID", "#2 ID"]].to_dict(orient='split')['data']
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(links)
    clusters = [list(c) for c in nx.connected_components(G)]
    new_links = []
    for c in clusters:
        for u, v in product(c, repeat=2):
            new_links.append(f"{u}++{v}")
    df_new: DataFrame = df.copy()
    df_new.loc[(df_new["#1 ID"] + "++" + df_new["#2 ID"]).isin(new_links), "# pred label"] = 1
    return df_new


standard_path = "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_input/test-large_wdoc_wtpc/dev.tsv"
standard_df = read_csv(standard_path, sep="\t")

print("before clustering:")
result = dict()
df = input_df
result["acc"] = accuracy_score(y_true=standard_df["Quality"], y_pred=df["# pred label"])
result["p"] = precision_score(y_true=standard_df["Quality"], y_pred=df["# pred label"])
result["r"] = recall_score(y_true=standard_df["Quality"], y_pred=df["# pred label"])
result["f1"] = f1_score(y_true=standard_df["Quality"], y_pred=df["# pred label"])
for k, v in result.items():
    print(f"{k}: {v}")

print("after clustering:")
result = dict()
df = connected_components_clustering(input_df)
result["acc"] = accuracy_score(y_true=standard_df["Quality"], y_pred=df["# pred label"])
result["p"] = precision_score(y_true=standard_df["Quality"], y_pred=df["# pred label"])
result["r"] = recall_score(y_true=standard_df["Quality"], y_pred=df["# pred label"])
result["f1"] = f1_score(y_true=standard_df["Quality"], y_pred=df["# pred label"])
for k, v in result.items():
    print(f"{k}: {v}")
