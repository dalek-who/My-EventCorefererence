#%%
import scipy.cluster.hierarchy as sch
import scipy as sp
import numpy as np
from pandas import DataFrame, Series, Index
from functools import partial

#%%
# Coreference关系
mention_ids = Series({"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7})
coref = [("a","b"), ("a","c"), ("c","d"), ("e","f"), ("f","g"), ("e","g"), ]
anti_mention_id = Series(mention_ids.index, index=mention_ids.values)  # 反向索引

#%%
# 距离矩阵
df = DataFrame(index=anti_mention_id.index, columns=anti_mention_id.index)
for i in anti_mention_id.index:
    for j in anti_mention_id.index:
        df.loc[i,j] = 0 if (anti_mention_id[i], anti_mention_id[j]) in coref or i==j else 1

def symmetric(X: np.array):
    return np.triu(X) + np.triu(X).T
distance_matrix = symmetric(df.values)
print(distance_matrix)

#%%
# 层次聚类
def distance(i,j):
    if (anti_mention_id[i], anti_mention_id[j]) in coref or i==j:
        return 0
    else:
        return 1
Z=sch.linkage(mention_ids.values, method='single', metric=distance)
f = sch.fcluster(Z, t=0.5)
