# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import argparse
import csv
import logging
import os
import random

import json
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from pandas import DataFrame, Series, read_csv
from math import ceil

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
try:
    from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear  # 新版本没有warmup_linear
except ImportError:
    from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from torch_geometric.data import DataLoader as GnnDataLoader

from configs import CONFIG
from utils.ExperimentRecordManager import generate_record, get_time_stamp, VisdomHelper
from utils.coref_eval import evaluate_coref_from_example
from models.clustering import connected_components_clustering, examples_to_predict_frame
from utils.DrawMetricsPicture import result_visualize
from utils.DrawClusterPicture import draw_cluster_graph

# 新加的
from preprocessing.Structurize.EcbClass import EcbPlusTopView
from preprocessing.Feature.MentionPair import InputFeaturesCreator
from models.DAST.DASTModel import *
from utils.VisdomHelper import EasyVisdom

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1_and_p_and_r(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    p = precision_score(y_true=labels, y_pred=preds)
    r = recall_score(y_true=labels, y_pred=preds)
    return {
        "precision": p,
        "recall": r,
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def store_model_to_checkpoint(model, output_checkpoint_dir):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_checkpoint_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(output_checkpoint_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())
    logger.info("Store checkpoint: model parameters to %s" % output_model_file)


def load_model_from_checkpoint(load_checkpoint_dir, num_labels, ModelClass):
    load_model_file = os.path.join(load_checkpoint_dir, WEIGHTS_NAME)
    load_config_file = os.path.join(load_checkpoint_dir, CONFIG_NAME)
    config = BertConfig(load_config_file)
    # model = BertForSequenceClassification(config, num_labels=num_labels)
    model = ModelClass(config, num_labels=num_labels)
    model.load_state_dict(torch.load(load_model_file))
    logger.info("Load checkpoint: model parameters from %s" % load_model_file)
    return model


def dataset_increase(examples: list, more: int, label: int=1):
    """
    扩增数据，解决正负例不平衡
    :param examples: 数据列表
    :param more: 对指定的label，再增加几倍的数据
    :param label: 需要扩增的例子的label
    :return: 扩增后的数据
    """
    sub_example = [ex for ex in examples if int(ex.label)==label]
    if more>=1:
        return examples + sub_example * more
    return examples

def dataset_shuffle(examples: list):
    """
    对数据集做shuffle
    :param examples:输入模型的数据集list
    :return: shuffle后的np.array
    """
    result = np.array(examples)
    np.random.shuffle(result)
    return result


def create_model_input_loader(feature_list: list, batch_size):
    all_BERT_input_ids = torch.tensor([f.BERT_input_ids for f in feature_list], dtype=torch.long)
    all_BERT_input_mask = torch.tensor([f.BERT_input_mask for f in feature_list], dtype=torch.long)
    all_BERT_segment_ids = torch.tensor([f.BERT_segment_ids for f in feature_list], dtype=torch.long)
    all_BERT_trigger_mask_a = torch.tensor([f.BERT_trigger_mask_a for f in feature_list], dtype=torch.long)
    all_BERT_trigger_mask_b = torch.tensor([f.BERT_trigger_mask_b for f in feature_list], dtype=torch.long)
    all_tfidf = torch.tensor([f.tfidf for f in feature_list], dtype=torch.float)
    all_label = torch.tensor([f.label for f in feature_list], dtype=torch.long)
    BERT_data = TensorDataset(
        all_BERT_input_ids, all_BERT_input_mask, all_BERT_segment_ids,
        all_BERT_trigger_mask_a, all_BERT_trigger_mask_b, all_tfidf, all_label)
    BERT_dataloader = DataLoader(BERT_data, batch_size=batch_size, shuffle=False)  # 之前已经手动shuffle

    all_gnn_datas = [f.gnn_data for f in feature_list]
    GNN_dataloader =GnnDataLoader(all_gnn_datas, batch_size=batch_size, shuffle=False)
    return BERT_dataloader, GNN_dataloader


def do_train(
        model, optimizer, warmup_linear, device, BERT_dataloader, GNN_dataloader, feature_list: list, n_gpu: int,
        fp16: bool,visdom_helper: EasyVisdom, num_train_optimization_steps: int, warmup_proportion: float,
        learning_rate: float, loss_each_step: dict, learning_rate_each_step: dict,
        use_document_feature: bool, use_sentence_trigger_feature: bool, use_argument_feature: bool, cross_document: bool):
    if model is not None:
        model.train()
    global train_global_step

    for step, batches in enumerate(tqdm(zip(BERT_dataloader, GNN_dataloader), desc="Train: ", total=len(BERT_dataloader))):
        train_global_step += 1
        # pytorch的dataloader和gnn的dataloader机制不同
        BERT_batch = tuple(t.to(device) for t in batches[0])
        BERT_input_ids, BERT_input_mask, BERT_segment_ids, BERT_trigger_mask_a, BERT_trigger_mask_b, tfidf, labels = BERT_batch
        GNN_batch = batches[1]
        GNN_batch.to(device, *('x', 'edge_index', 'edge_weight', 'trigger_mask', 'y'))
        GNN_x, GNN_edge_index, GNN_edge_weight, GNN_trigger_mask, GNN_labels = \
            GNN_batch.x, GNN_batch.edge_index, GNN_batch.edge_weight, GNN_batch.trigger_mask, GNN_batch.y
        logits = model(
            device,
            BERT_input_ids, BERT_segment_ids, BERT_input_mask, BERT_trigger_mask_a, BERT_trigger_mask_b,
            tfidf,
            GNN_x, GNN_edge_index, GNN_edge_weight, GNN_trigger_mask,
            use_document_feature, use_sentence_trigger_feature, use_argument_feature, cross_document)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        # 学习率曲线
        try:
            lr_this_step = learning_rate * warmup_linear(train_global_step / num_train_optimization_steps, warmup_proportion)
        except:
            lr_this_step = learning_rate * warmup_linear.get_lr(train_global_step, warmup_proportion)
            # lr_this_step = learning_rate * warmup_linear.get_lr(train_global_step / num_train_optimization_steps, warmup_proportion)

        # 更新
        if fp16:
            # modify learning rate with special warm up BERT uses
            # if args.fp16 is False, BertAdam is used that handles this automatically
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
        optimizer.step()
        optimizer.zero_grad()

        # 记录数据点
        loss_each_step[train_global_step] = loss.detach().cpu().item()
        learning_rate_each_step[train_global_step] = lr_this_step
        # 绘图
        draw_visdom_each_step(
            visdom_helper=visdom_helper, step=train_global_step,
            train_loss=loss_each_step[train_global_step], learning_rate=learning_rate_each_step[train_global_step])


def do_eval(model, device, BERT_dataloader, GNN_dataloader, feature_list: list,
            use_document_feature: bool, use_sentence_trigger_feature: bool, use_argument_feature: bool, cross_document: bool):
    if model is not None:
        model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for step, batches in enumerate(tqdm(zip(BERT_dataloader, GNN_dataloader), desc="Eval: ", total=len(BERT_dataloader))):
        # pytorch的dataloader和gnn的dataloader机制不同
        BERT_batch = tuple(t.to(device) for t in batches[0])
        BERT_input_ids, BERT_input_mask, BERT_segment_ids, BERT_trigger_mask_a, BERT_trigger_mask_b, tfidf, labels = BERT_batch
        GNN_batch = batches[1]
        GNN_batch.to(device, *('x', 'edge_index', 'edge_weight', 'trigger_mask', 'y'))
        GNN_x, GNN_edge_index, GNN_edge_weight, GNN_trigger_mask, GNN_labels = \
            GNN_batch.x, GNN_batch.edge_index, GNN_batch.edge_weight, GNN_batch.trigger_mask, GNN_batch.y
        with torch.no_grad():
            logits = model(
                device,
                BERT_input_ids, BERT_segment_ids, BERT_input_mask, BERT_trigger_mask_a, BERT_trigger_mask_b,
                tfidf,
                GNN_x, GNN_edge_index, GNN_edge_weight, GNN_trigger_mask,
                use_document_feature, use_sentence_trigger_feature, use_argument_feature, cross_document)

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    # 计算评价指标
    preds = np.argmax(preds[0], axis=1)
    labels = np.array([f.label for f in feature_list])
    eval_result_classifier = acc_and_f1_and_p_and_r(preds, labels)  # 分类器指标
    eval_result_coref = evaluate_coref_from_example(feature_list, preds)  # coref指标
    result = dict(eval_result_classifier, **eval_result_coref)
    result['eval_loss']  = eval_loss / nb_eval_steps
    return result, preds


def prepare_optimizer(model, fp16: bool, loss_scale: float, learning_rate: float, warmup_proportion: float, num_train_optimization_steps: int):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters, lr=learning_rate, bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
    try:
        global warmup_linear
        warmup_linear = warmup_linear
    except NameError:
        warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion, t_total=num_train_optimization_steps)

    return optimizer, warmup_linear


def compare_with_checkpoint(model, output_checkpoint_dir, logger, compare_dict: dict, eval_result: dict,
                            epoch: int, global_step: int, by_what: str="CoNLL_f1"):
    if eval_result[by_what] > compare_dict["best_result"]:
        logger.info(f"********** best model at Epoch {epoch} with {by_what}={eval_result[by_what]}, update checkpoint **********")
        for k, v in eval_result.items():
            logger.info("%s = %s" % (k, v))
        compare_dict["best_result"] = eval_result[by_what]
        compare_dict["best_epoch"] = epoch
        compare_dict["best_global_step"] = global_step
        store_model_to_checkpoint(model=model, output_checkpoint_dir=output_checkpoint_dir)


def draw_visdom_each_epoch(visdom_helper: EasyVisdom, epoch: int, train_result: dict, eval_result: dict):

    # 每个指标一张图, 图里有train和dev两条线
    for which_line in ("MUC_f1", "B-cubed_f1", "CEAFe_f1", "CEAFm_f1", "BLANCc_f1", "BLANCn_f1", "BLANC_f1",
                       "CoNLL_f1", "AVG_f1"):
        for name, result in [("Train", train_result), ("Eval", eval_result)]:
            visdom_helper.update_line(
                x_axis_name="Epoch", y_axis_name="Evaluation Result", line_name="%s_%s" % (name, which_line),
                title_name="%s of each Epoch" % which_line, x=epoch, y=result[which_line])

    # 一张loss图，里面有train和dev两条线
    for name, result in [("Train", train_result), ("Eval", eval_result)]:
        visdom_helper.update_line(
            x_axis_name="Epoch", y_axis_name="Loss", line_name="%s_loss" % name,
            title_name="Loss of each Epoch", x=epoch, y=result["eval_loss"])

    # dev，train各一张图，每张图里每个单项指标一条线
    for which_line in ("MUC_f1", "B-cubed_f1", "CEAFe_f1", "BLANC_f1"):
        for name, result in [("Train", train_result), ("Eval", eval_result)]:
            visdom_helper.update_line(x_axis_name="Epoch", y_axis_name="Evaluation Result", line_name="%s_%s" % (name, which_line),
                                      title_name="%s Single Metrics of each Epoch" % name, x=epoch, y=result[which_line])

    # dev，train各一张图，每张图里每个复合指标一条线
    for which_line in ("BLANC_f1", "CoNLL_f1", "AVG_f1"):
        for name, result in [("Train", train_result), ("Eval", eval_result)]:
            visdom_helper.update_line(x_axis_name="Epoch", y_axis_name="Evaluation Result", line_name="%s_%s" % (name, which_line),
                                      title_name="%s Composite Metrics of each epoch" % name, x=epoch, y=result[which_line])


def draw_visdom_each_step(visdom_helper: EasyVisdom, step: int, train_loss: float, learning_rate: float):
    # loss
    visdom_helper.update_line(x_axis_name="Step", y_axis_name="Loss", line_name="Train_Loss",
                              title_name="Train Loss of each Step", x=step, y=train_loss)
    # 50 step平滑loss
    visdom_helper.update_line_smooth(x_axis_name="Step", y_axis_name="Loss", line_name="Train_Loss",
                                     title_name="Train Loss of each 50 Step", x=step, y=train_loss, buf_size=50)
    # loss
    visdom_helper.update_line_smooth(x_axis_name="Step", y_axis_name="Loss", line_name="Train_Loss",
                                     title_name="Train Loss of each 100 Step", x=step, y=train_loss, buf_size=100)
    # 学习率
    visdom_helper.update_line(x_axis_name="Step", y_axis_name="Learning Rate", line_name="Learning_Rate",
                              title_name="Learning Rate", x=step, y=learning_rate)


def input_feature_statistics(feature_list: list):
    """
    统计数据集的一些情况
    :param examples: 数据列表
    :return: 统计得到的字典
    """
    stat_result = dict()
    stat_result["data_number"] = len(feature_list)  # 数据数量
    stat_result["pos_example_number"] = len([f for f in feature_list if int(f.label)==1])  # 正例数量
    stat_result["pos_rate"] = stat_result["pos_example_number"] / stat_result["data_number"] if stat_result["data_number"]!=0 else 0   # 正例比例
    stat_result["neg_example_number"] = stat_result["data_number"] - stat_result["pos_example_number"]
    stat_result["neg_rate"] = 1 - stat_result["pos_rate"]
    return stat_result


train_global_step = 0
def my_main():
    """
    定义main的命令行参数args
    :return: args
    """
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--trigger_half_window",
                        default=2,
                        type=int,
                        help="Context words number before and after trigger word.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict_only",
                        action='store_true',
                        help="Whether to run predict on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--load_trained",
                        action='store_true',
                        help="Whether to load fine-turned model.")
    parser.add_argument("--load_model_dir",
                        default="",
                        type=str,
                        help="Directory where do you want to load the fine-turned model parameters")
    parser.add_argument("--predict_output_dir",
                        default="",
                        type=str,
                        help="Directory where do you want to store the predict result")
    parser.add_argument("--train_output_dir",
                        default="",
                        type=str,
                        help="The output directory for train where the model predictions and checkpoints will be written.")
    parser.add_argument("--coref_level",
                        required=True,
                        type=str,
                        help="Which level's coreference, could be 'within_document', 'cross_document', 'cross_topic'")
    parser.add_argument("--description",
                        default="",
                        type=str,
                        help="Description of this experiment")
    parser.add_argument("--increase_positive",
                        default=0,
                        type=int,
                        help="How many times positive examples should be increased.")
    parser.add_argument("--draw_visdom",
                        action="store_true",
                        help="Weather to use visdom to draw.")
    parser.add_argument("--use_document_feature",
                        action="store_true",
                        help="Whether to use document features.")
    parser.add_argument("--use_sentence_trigger_feature",
                        action="store_true",
                        help="Whether to use sentence and trigger features")
    parser.add_argument("--use_argument_feature",
                        action="store_true",
                        help="Whether to use argument features")

    args = parser.parse_args()

    # 运算设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))


    # 随机种子。如果每次设定的随机种子一样，则随机数的序列也一样
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 判断输出目录是否已存在且非空
    if not args.do_train and not args.do_predict_only:
        raise ValueError("At least one of `do_train` or `do_predict_only` must be True.")
    # 模型参数目录
    if os.path.exists(args.train_output_dir) and os.listdir(args.train_output_dir) and args.do_train:
        raise ValueError("Train output directory ({}) already exists and is not empty.".format(args.train_output_dir))
    if args.do_train and not os.path.exists(args.train_output_dir):
        os.makedirs(args.train_output_dir)
    # 预测结果目录
    if os.path.exists(args.predict_output_dir) and os.listdir(args.predict_output_dir):
        raise ValueError("Predict output directory ({}) already exists and is not empty.".format(args.train_output_dir))
    if not os.path.exists(args.predict_output_dir):
        os.makedirs(args.predict_output_dir)

    # 准备模型, 先准备模型后准备数据，防止花很长时间与处理数据后模型却是NoneType
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_-1')
    # if args.use_sentence_trigger_feature and not args.use_document_feature and not args.use_argument_feature:
    #     ModelClass = DAST_SentenceTrigger
    # elif not args.use_sentence_trigger_feature and not args.use_document_feature and args.use_argument_feature:
    #     ModelClass = DAST_Argument
    # else:
    #     ModelClass = None
    ModelClass = DAST

    if ModelClass is not None and not args.load_trained:
        model = ModelClass.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=2)
    elif ModelClass is not None and args.load_trained:
        model = load_model_from_checkpoint(load_checkpoint_dir=args.load_model_dir, num_labels=2, ModelClass=ModelClass)
    else:
        model = None
        raise ModuleNotFoundError("Please choose a model")

    if args.fp16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)



    # 是否跨文档，是否跨主题
    if args.coref_level == "within_document":
        cross_document, cross_topic = False, False
    elif args.coref_level == "cross_document":
        cross_document, cross_topic = True, False
    elif args.coref_level == "cross_topic":
        cross_document, cross_topic = True, True
    else:
        raise ValueError("--coref_level=%s is invalid" % args.coref_level)

    # 准备数据
    dataset_statistics_dict = dict()  # 统计数据集
    try:
        ECBplus = EcbPlusTopView.load()  # 预先存储好，load比较快
    except :
        ECBplus = EcbPlusTopView()
        ECBplus.dump()
    feature_creator = InputFeaturesCreator(ECBPlus=ECBplus)
    try:
        dev_features = feature_creator.load(
            topics=CONFIG.topics_test, cross_document=cross_document, cross_topic=cross_topic, ignore_order=True,
            max_seq_length=args.max_seq_length, trigger_half_window=args.trigger_half_window)
        logger.info("*******  Load eval features cache ******")
    except:
        logger.info("*******  Create eval features  ******")
        dev_features = feature_creator.create_from_dataset(
            topics=CONFIG.topics_test, cross_document=cross_document, cross_topic=cross_topic, ignore_order=True,
            max_seq_length=args.max_seq_length, trigger_half_window=args.trigger_half_window)
    dataset_statistics_dict["dev"] = input_feature_statistics(dev_features)

    num_train_optimization_steps = 0
    train_features = []
    if args.do_train:
        try:
            train_features = feature_creator.load(
                topics=CONFIG.topics_train, cross_document=cross_document, cross_topic=cross_topic, ignore_order=True,
                max_seq_length=args.max_seq_length, trigger_half_window=args.trigger_half_window)
            logger.info("*******  Load train features cache ******")
        except:
            logger.info("*******  Create train features  ******")
            train_features = feature_creator.create_from_dataset(
                topics=CONFIG.topics_train, cross_document=cross_document, cross_topic=cross_topic, ignore_order=True,
                max_seq_length=args.max_seq_length, trigger_half_window=args.trigger_half_window)
        dataset_statistics_dict["train"] = input_feature_statistics(train_features)
        dataset_statistics_dict["train"]["positive_increase"] = args.increase_positive
        if args.increase_positive >= 1:  # 扩增正例
            train_features = dataset_increase(examples=train_features, more=args.increase_positive, label=1)
        train_features = dataset_shuffle(train_features)
        num_train_optimization_steps = ceil(len(train_features) / args.train_batch_size) * args.num_train_epochs


    # 准备绘图
    time_stamp = get_time_stamp()
    visdom_helper = EasyVisdom(env_name=f"DAST env: %s %s " % (time_stamp, args.description), enable=args.draw_visdom)

    eval_BERT_dataloader_dev, eval_GNN_dataloader_dev = create_model_input_loader(feature_list=dev_features, batch_size=args.eval_batch_size)
    eval_BERT_dataloader_train, eval_GNN_dataloader_train = create_model_input_loader(feature_list=train_features, batch_size=args.eval_batch_size)
    visdom_helper.show_dict(title_name="Command Line Arguments", dic=vars(args))
    visdom_helper.show_dict(title_name="Eval Data Statistics", dic=input_feature_statistics(dev_features))
    if args.do_train:
        # 准备优化器
        optimizer, warmup_linear = prepare_optimizer(
            model, fp16=args.fp16, loss_scale=args.loss_scale, learning_rate=args.learning_rate,
            warmup_proportion=args.warmup_proportion, num_train_optimization_steps=num_train_optimization_steps)
        # train数据
        train_BERT_dataloader, train_GNN_dataloader = create_model_input_loader(feature_list=train_features, batch_size=args.train_batch_size)
        visdom_helper.show_dict(title_name="Train Data Statistics", dic=input_feature_statistics(train_features))


        global train_global_step
        compare_dict = {"best_result": -2, "best_epoch": -2, "best_global_step": -2}
        train_curve_datas = {
            "train_loss_each_step": dict(),
            "learning_rate_each_step": dict(),
            "eval_result_each_epoch_on_train": dict(),
            "eval_result_each_epoch_on_dev": dict(),
        }

        # 训练开始前先eval一次
        train_result, train_pred = do_eval(model=model, device=device, BERT_dataloader=eval_BERT_dataloader_train,
                                           GNN_dataloader=eval_GNN_dataloader_train, feature_list=train_features,
                                           use_argument_feature=args.use_argument_feature,
                                           use_sentence_trigger_feature=args.use_sentence_trigger_feature,
                                           use_document_feature=args.use_document_feature,
                                           cross_document=cross_document)

        eval_result, eval_pred = do_eval(model=model, device=device, BERT_dataloader=eval_BERT_dataloader_dev,
                                         GNN_dataloader=eval_GNN_dataloader_dev, feature_list=dev_features,
                                         use_argument_feature=args.use_argument_feature,
                                         use_sentence_trigger_feature=args.use_sentence_trigger_feature,
                                         use_document_feature=args.use_document_feature,
                                         cross_document=cross_document)
        train_curve_datas["eval_result_each_epoch_on_train"][0] = train_result
        train_curve_datas["eval_result_each_epoch_on_dev"][0] = eval_result
        compare_with_checkpoint(
            model=model, output_checkpoint_dir=args.train_output_dir, logger=logger, compare_dict=compare_dict,
            eval_result=eval_result, epoch=0, global_step=0, by_what=CONFIG.CHECKPOINT_BY_WHAT)
        draw_visdom_each_epoch(visdom_helper=visdom_helper, epoch=0, train_result=train_result, eval_result=eval_result)
        # 训练
        for epoch in trange(1, int(args.num_train_epochs)+1, desc="Epoch"):
            do_train(
                model=model, optimizer=optimizer, warmup_linear=warmup_linear, device=device,
                BERT_dataloader=train_BERT_dataloader, GNN_dataloader=train_GNN_dataloader,
                feature_list=train_features, n_gpu=n_gpu, fp16=args.fp16, visdom_helper=visdom_helper,
                num_train_optimization_steps=num_train_optimization_steps, warmup_proportion=args.warmup_proportion,
                learning_rate=args.learning_rate, loss_each_step=train_curve_datas["train_loss_each_step"],
                learning_rate_each_step=train_curve_datas["learning_rate_each_step"],
                use_argument_feature=args.use_argument_feature,
                use_sentence_trigger_feature=args.use_sentence_trigger_feature,
                use_document_feature=args.use_document_feature,
                cross_document=cross_document)
            train_result, train_pred = do_eval(model=model, device=device, BERT_dataloader=eval_BERT_dataloader_train,
                                               GNN_dataloader=eval_GNN_dataloader_train, feature_list=train_features,
                                               use_argument_feature=args.use_argument_feature,
                                               use_sentence_trigger_feature=args.use_sentence_trigger_feature,
                                               use_document_feature=args.use_document_feature,
                                               cross_document=cross_document)
            eval_result, eval_pred = do_eval(model=model, device=device, BERT_dataloader=eval_BERT_dataloader_dev,
                                             GNN_dataloader=eval_GNN_dataloader_dev, feature_list=dev_features,
                                             use_argument_feature=args.use_argument_feature,
                                             use_sentence_trigger_feature=args.use_sentence_trigger_feature,
                                             use_document_feature=args.use_document_feature,
                                             cross_document=cross_document)
            train_curve_datas["eval_result_each_epoch_on_train"][epoch] = train_result
            train_curve_datas["eval_result_each_epoch_on_dev"][epoch] = eval_result
            compare_with_checkpoint(
                model=model, output_checkpoint_dir=args.train_output_dir, logger=logger, compare_dict=compare_dict,
                eval_result=eval_result, epoch=epoch, global_step=train_global_step, by_what=CONFIG.CHECKPOINT_BY_WHAT)
            draw_visdom_each_epoch(visdom_helper=visdom_helper, epoch=epoch, train_result=train_result, eval_result=eval_result)
        # 保存训练曲线的所有数据点,包括每个step的loss和学习率，每个epoch的测评
        best_eval_result = train_curve_datas["eval_result_each_epoch_on_dev"][compare_dict["best_epoch"]]
        with open(os.path.join(args.train_output_dir, CONFIG.TRAIN_CURVE_DATA_FILE_NAME), "w") as f:
            json.dump(train_curve_datas, f)  # 学习率list点太多，不自动缩进了
        # 保存最好结果
        with open(os.path.join(args.train_output_dir, CONFIG.TRAIN_BEST_EVAL_RESULT_FILE_NAME), "w") as f:
            json.dump(best_eval_result, f, indent=4)
        # 保存对数据集的统计
        with open(os.path.join(args.train_output_dir, CONFIG.DATASET_STATISTICS_FILE_NAME), "w") as f:
            json.dump(dataset_statistics_dict, f, indent=4)
        # 保存命令行参数选项
        with open(os.path.join(args.train_output_dir, CONFIG.COMMAND_LINE_ARGUMENTS_FILE_NAME), "w") as f:
            json.dump(vars(args), f, indent=4)
        # 生成曲线图
        result_visualize(curve_data_dict=train_curve_datas, output_dir=args.train_output_dir)
        # 展示最好结果
        visdom_helper.show_dict(title_name="Best Eval Result", dic=best_eval_result)
        # load最好的模型
        model = load_model_from_checkpoint(load_checkpoint_dir=args.train_output_dir, num_labels=2, ModelClass=ModelClass)

    if args.do_predict_only:
        eval_result, eval_pred = do_eval(model=model, device=device, BERT_dataloader=eval_BERT_dataloader_dev,
                                         GNN_dataloader=eval_GNN_dataloader_dev, feature_list=dev_features,
                                         use_argument_feature=args.use_argument_feature,
                                         use_sentence_trigger_feature=args.use_sentence_trigger_feature,
                                         use_document_feature=args.use_document_feature,
                                         cross_document=cross_document)
        visdom_helper.show_dict(title_name="Eval Result", dic=eval_result)
        draw_cluster_graph(dev_features, eval_pred)


if __name__ == "__main__":
    # main()
    my_main()
