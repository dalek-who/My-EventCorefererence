import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data as GraphData, Batch as GraphBatch, DataLoader as GraphDataLoader

import random

seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class GData(object):
    def __init__(self, node_list):
        self.node_list = node_list

class Feature(object):
    def __init__(self, sentence_mask, id, gdata, label):
        self.sentence_mask = sentence_mask
        self.id = id
        self.gdata = gdata
        self.label = label

class DASTDataset(Dataset):
    def __init__(self, fearure_list:list):
        self.feature_list = fearure_list

    def __getitem__(self, item):
        id = self.feature_list[item].id
        sentence_mask = torch.Tensor(self.feature_list[item].sentence_mask)
        gdata = tuple([self.feature_list[item].gdata,])
        label = self.feature_list[item].label
        return id, sentence_mask, 0, label

    def __len__(self):
        return len(self.feature_list)

seq_len = 5
batch_size = 3
data_num = 17
datas = []
for i in range(data_num):
    sentence_mask = [random.choice([0,1]) for _ in range(seq_len)]
    gdata = GData(list(range(i)))
    id = i
    label = random.choice([0,1])
    datas.append(Feature(sentence_mask=sentence_mask, id=id, gdata=0, label=label))

gdatas = []
for i in range(data_num):
    x = torch.Tensor([range(i)]).t()
    id = i
    gdata = GraphData(x=x)
    gdata.id = id
    gdatas.append(gdata)

dataset = DASTDataset(datas)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
gloader = GraphDataLoader(dataset=gdatas, batch_size=batch_size, shuffle=False)

for batch, gbatch in zip(loader, gloader):
    id, sentence_mask, gdata, label = batch
    sentence_mask