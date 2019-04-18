#%%
from visdom import Visdom
import numpy as np
from time import sleep
import torch
from matplotlib import pyplot as plt
from pandas import Series

#%%
viz = Visdom(env='visdom_test')
round,loss=0,0
win = viz.line(
    X=np.array([round]),
    Y=np.array([loss]),
    opts=dict(title='train loss'))

round = 0
for epoch in range(3):
    for batch in range(3):
        round += 1
        loss = np.exp(-round)*100 + np.random.randn()
        loss = torch.Tensor([loss])
        viz.line(
            X=np.array([round]),
            Y=loss,
            win=win,#win要保持一致
            update='append')
        sleep(0.5)

args = {"learning-rate": 0.01, "--fp16": False}
text = "     \n".join(["%s: %s" % (k,v) for k,v in args.items()])
text_win = viz.text(text="")
for k,v in args.items():
    viz.text(text="%s: %s" % (k,v), win=text_win, append=True)
