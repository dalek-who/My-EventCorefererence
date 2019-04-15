import sys
sys.path.append("../..")
import csv
from pandas import DataFrame, Series, read_csv
import os
import time
from configs import CONFIG


def generate_record(table_name: str,
                coref_level: str,
                input_data_path: str,
                model_parameter_path: str,
                output_evaluate_dir: str=None,
                eval_result_dict: dict=None,
                args_dict: dict=None,
                description: str="") -> Series:
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
    record["time_stamp"] = time.strftime('%Y.%m.%d-%H:%M:%S',time.localtime(time.time()))
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
    header = not os.path.exists(table_path)
    record.to_frame().T.to_csv(table_path, sep="\t", index=False, mode="a", header=header)
    return record
