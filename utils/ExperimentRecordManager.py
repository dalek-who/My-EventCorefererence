import sys
sys.path.append("../..")
import csv
from pandas import DataFrame, Series, read_csv
import os
import time
from configs import CONFIG
from visdom import Visdom
import numpy as np
from collections import defaultdict, deque

def get_time_stamp():
    return time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime(time.time()))

def generate_record(
        table_name: str,
        coref_level: str,
        input_data_path: str,
        model_parameter_path: str,
        output_evaluate_dir: str=None,
        eval_result_dict: dict=None,
        args_dict: dict=None,
        description: str="",
        time_stamp: str=None) -> Series:
    """
    :param table_name: 表的名字
    :param coref_level: "within_document"：篇章内, "within_topic"：话题内, "global"：全局
    :param input_data_path: 模型输入数据的路径
    :param model_parameter_path: 模型参数所在目录
    :param output_evaluate_dir: 模型评估结果/预测结果所在目录
    :param eval_result_dict: 评价指标结果
    :param args_dict: 运行此模型的命令行参数
    :param description: 对此次实验的注释
    :return:
    """
    # 生成record
    record = Series()
    record["time_stamp"] = time_stamp if time_stamp is not None else time.strftime('%Y.%m.%d-%H:%M:%S',time.localtime(time.time()))
    record["coref_level"] = coref_level
    record["input_data_path"] = os.path.abspath(input_data_path)
    record["model_parameter_path"] = os.path.abspath(model_parameter_path)
    if output_evaluate_dir is not None:
        record["output_evaluate_dir"] = os.path.abspath(output_evaluate_dir)
    if eval_result_dict is not None:
        for k,v in eval_result_dict.items():
            record[k] = v
    if args_dict is not None:
        for k, v in args_dict.items():
            record[k] = v
    record["description"] = description

    # 读取并追加record,写回tsv
    if table_name.endswith((".csv", ".tsv")):
        table_name = table_name[:-4]
    table_path = os.path.join(CONFIG.EXPERIMENT_RECOED_DIR, table_name + ".tsv")
    # try:
    #     table = read_csv(table_path, sep="\t", quoting=csv.QUOTE_NONE)
    # except FileNotFoundError:
    #     table = DataFrame()
    # table = table.append(record, ignore_index=True)
    # table.to_csv(table_path, sep="\t", index=False)
    if not os.path.exists(os.path.dirname(table_path)):
        os.makedirs(os.path.dirname(table_path))
    header = not os.path.exists(table_path)
    record.to_frame().T.to_csv(table_path, sep="\t", index=False, mode="a", header=header)
    return record


class VisdomHelper(object):
    def __init__(self, env: str, enable: bool=True):
        self.enable = enable  # 是否画图
        if not self.enable:
            return
        self.viz = Visdom(env=env)
        self.win = {"train_loss": None,
                    "train_acc": None,
                    "eval_loss": None,
                    "eval_acc": None,
                    "args": None,
                    "eval": None}
        # mini-batch存在loss抖动的问题，可以几个batch求个平均loss
        self.smooth_buffer_dict = {}  # 把前n次的loss求平均进行平滑
        self.average_buffer_dict = {}  # 每n次求一次平均loss

    def update_line(self, line_name: str, round: int, value: float):
        if not self.enable:
            return
        if self.win.get(line_name) is None:
            self.win[line_name] = self.viz.line(
                X=np.array([0]),
                Y=np.array([0]),
                opts=dict(title=line_name)
            )
        self.viz.line(
            X=np.array([round]),
            Y=np.array([value]),
            win=self.win[line_name],  # win要保持一致
            update='append')

    def update_scatter(self, line_name: str, round: int, value: float):
        if not self.enable:
            return
        if self.win.get(line_name) is None:
            self.win[line_name] = self.viz.scatter(
                X=np.array([[0, 0]]),
                # Y=np.array([0]),  # 散点图的X是一堆点坐标，Y是颜色
                opts=dict(title=line_name,
                          markersize=1)
            )
        self.viz.scatter(
            X=np.array([[round, value]]),
            # Y=np.array([value]),
            win=self.win[line_name],
            update='append')

    def show_dict(self, text_name: str, dic: dict):
        if not self.enable:
            return
        if self.win.get(text_name) is None:
            self.win[text_name] = self.viz.text(
                text="",
                opts=dict(title=text_name))
        for k, v in dic.items():
            self.viz.text(
                text="%s: %s" % (k, v),
                win=self.win[text_name],
                append=True
            )

    def update_line_average(self, line_name: str, round: int, value: float, buf_size: int):
        """
        每调用此函数buf_size次画一个点，该点的值是buf_size个点的平均
        :param line_name: 线名
        :param round: 第几次调用此函数
        :param value: 此次的值
        :param buf_size: 存储有待于平均的点的buffer长度
        :return: None
        """
        if not self.enable:
            return
        if self.win.get(line_name) is None:
            self.win[line_name] = self.viz.line(
                X=np.array([0]),
                Y=np.array([0]),
                opts=dict(title=line_name)
            )
        buf_id = "%s-%s" % (line_name, buf_size)
        if self.average_buffer_dict.get(buf_id) is None:
            self.average_buffer_dict[buf_id] = deque(maxlen=buf_size)
        buf = self.average_buffer_dict[buf_id]
        buf.append(value)
        if len(buf) == buf.maxlen:
            self.viz.line(
                X=np.array([round]),
                Y=np.array([np.average(buf)]),
                win=self.win[line_name],
                update='append')
            buf.clear()

    def update_line_smooth(self, line_name: str, round: int, value: float, buf_size: int):
        """
        每次画这个点以及前buf_size个点的平均值进行平滑
        与update_line_average的区别是update_line_average每buf_size次才画一个点，
        而update_line_smooth每次都会画一个平滑后的点
        :param line_name: 线名
        :param round: 第几次调用此函数
        :param value: 此次的值
        :param buf_size: 存储有待于平均的点的buffer长度
        :return: None
        """

        if not self.enable:
            return
        if self.win.get(line_name) is None:
            self.win[line_name] = self.viz.line(
                X=np.array([0]),
                Y=np.array([0]),
                opts=dict(title=line_name)
            )
        buf_id = "%s-%s" % (line_name, buf_size)
        if self.smooth_buffer_dict.get(buf_id) is None:
            self.smooth_buffer_dict[buf_id] = deque(maxlen=buf_size)
        buf = self.smooth_buffer_dict[buf_id]
        buf.append(value)
        self.viz.line(
            X=np.array([round]),
            Y=np.array([np.average(buf)]),
            win=self.win[line_name],
            update='append')

    def update_line_total_average(self, line_name: str, round: int, value: float):
        if not self.enable:
            return
        if self.win.get(line_name) is None:
            self.win[line_name] = self.viz.line(
                X=np.array([0]),
                Y=np.array([0]),
                opts=dict(title=line_name)
            )
        buf_id = line_name
        if self.average_buffer_dict.get(buf_id) is None:
            self.average_buffer_dict[buf_id] = deque()
        buf = self.average_buffer_dict[buf_id]
        buf.append(value)
        self.viz.line(
            X=np.array([round]),
            Y=np.array([np.average(buf)]),
            win=self.win[line_name],
            update='append')
