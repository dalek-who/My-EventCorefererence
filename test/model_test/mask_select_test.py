#%%
from torch import Tensor
import torch

#%%
vertex_embedding = [
    [1, 2, 3, 4, 5],
    [11, 12, 13, 14, 15],
    [21, 22, 23, 24, 25],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45],
    [51, 52, 53, 54, 55],
    [61, 62, 63, 64, 65],
    [71, 72, 73, 74, 75],
    [81, 82, 83, 84, 85],
    [91, 92, 93, 94, 95],
]

mask = [0,0,1,0,1,0,0,1,1,0]

ts_embedding = Tensor(vertex_embedding)
ts_mask = Tensor(mask).unsqueeze(-1).type(torch.uint8)  # mask必须为unit8型。

#%%
selected = ts_embedding.masked_select(ts_mask)
trigger_vertex_embedding = selected.view(-1, 2, 5)  # view: (-1是batch_size此处自动求解, 2表示batch里每个图有两个trigger点, 5代表每个点embedding的维度)