from visdom import Visdom
import numpy as np
from collections import deque

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', enable=True):
        self.enable = enable
        if not self.enable:
            return
        self.viz = Visdom()
        self.env = env_name
        self.tables = {}
        # mini-batch存在loss抖动的问题，可以几个batch求个平均loss
        self.buffer_dict = {}  # 对loss进行平滑

    def update_line(self, x_axis_name: str, y_axis_name: str, line_name: str, title_name:str, x: float, y: float):
        """
        在图中画上一个点。会自动把名为line_name的线放到名为title_name的图里，自动用不同颜色区分
        :param x_axis_name: x轴名称
        :param y_axis_name: y轴名称
        :param line_name: 画的线的名称
        :param title_name: 图的标题
        :param x: x值
        :param y: y值
        :return: None
        """
        if not self.enable:
            return
        if title_name not in self.tables:
            self.tables[title_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[line_name],
                    title=title_name,
                    xlabel=x_axis_name,
                    ylabel=y_axis_name)
            )
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]),
                          env=self.env, win=self.tables[title_name], name=line_name, update ='append')


    def update_line_smooth(self, x_axis_name: str, y_axis_name: str, line_name: str, title_name: str,
                           x: float, y: float, buf_size: int):
        """
        画线时，每次画这个点以及前buf_size个点的平均值进行平滑
        :param x_axis_name: x轴名称
        :param y_axis_name: y轴名称
        :param line_name: 画的线的名称
        :param title_name: 图的标题
        :param x: x值
        :param y: y值
        :param buf_size: 存储有待于平均的点的buffer长度
        :return: None
        """

        if not self.enable:
            return
        buf_id = "%s-%s-%s" % (title_name, line_name, buf_size)
        if self.buffer_dict.get(buf_id) is None:
            self.buffer_dict[buf_id] = deque(maxlen=buf_size)
        buf = self.buffer_dict[buf_id]
        buf.append(y)
        draw_x = x
        draw_y = np.average(buf)
        self.update_line(x_axis_name=x_axis_name, y_axis_name=y_axis_name, line_name=line_name, title_name=title_name,
                         x=draw_x, y=draw_y)

    def show_dict(self, title_name: str, dic: dict):
        if not self.enable:
            return
        if title_name not in self.tables:
            self.tables[title_name] = self.viz.text(
                text="",
                opts=dict(title=title_name))
        for k, v in dic.items():
            self.viz.text(
                text="%s: %s" % (k, v),
                win=self.tables[title_name],
                append=True
            )


# plotter = VisdomLinePlotter()
# train_loss = lambda x: x**2
# eval_loss = lambda x: 1.5*x**2
#
# from time import sleep
# for step in range(1,101):
#     plotter.update_line(
#         x_axis_name="step", y_axis_name="loss", line_name="train loss", title_name="Loss in training",
#         x=step, y=train_loss(step))
#     plotter.update_line(
#         x_axis_name="step", y_axis_name="loss", line_name="eval loss", title_name="Loss in training",
#         x=step,y=eval_loss(step))
#     plotter.update_line_smooth(
#         x_axis_name="step", y_axis_name="loss xxx", line_name="smooth eval loss", title_name="Loss in training",
#         x=step,y=eval_loss(step), buf_size=5)
#     sleep(0.2)
# plotter.show_dict(title_name="args", dic={"lr":0.01, "data_dir": "/hello/", "do_train": True})
