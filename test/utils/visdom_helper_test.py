from time import sleep
from utils.ExperimentRecordManager import VisdomHelper, get_time_stamp
import torch
import numpy as np

time_stamp = get_time_stamp()
vh = VisdomHelper(env="visdom test " + time_stamp)
args = {"train_input": "./input/train.tsv", "lr": 0.01, "fp16": True}

# 画损失函数
round = 0
for epoch in range(3):
    for batch in range(5):
        round += 1
        loss = np.exp(-round)*100 + np.random.randn()
        loss = torch.Tensor([loss])
        vh.update_line(line_name="train_loss", round=round, value=loss)
        sleep(0.5)

# 在visdom里打印字典
vh.show_dict(text_name="args", dic=args)
