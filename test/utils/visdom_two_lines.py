import numpy as np
from time import sleep
from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', enable: bool=True):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        self.smooth_buffer_dict = {}  # 把前n次的loss求平均进行平滑
        self.enabel = enable

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

plotter = VisdomLinePlotter()
train_loss = lambda x: x**2
eval_loss = lambda x: 1/x

for epoch in range(1,101):
    plotter.plot(var_name="loss", split_name="train loss", title_name="Loss in training", x=epoch, y=train_loss(epoch))
    plotter.plot(var_name="loss", split_name="dev loss", title_name="Loss in training", x=epoch, y=eval_loss(epoch))
    sleep(0.2)