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

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear  # 新版本没有warmup_linear
# from pytorch_pretrained_bert.optimization import BertAdam

from configs import CONFIG
from utils.ExperimentRecordManager import generate_record, get_time_stamp, VisdomHelper
from utils.coref_eval import evaluate_coref_from_example
from models.clustering import connected_components_clustering, examples_to_predict_frame

# 新加的
from preprocessing.Structurize.EcbClass import EcbPlusTopView
from preprocessing.Feature.MentionPair import InputFeaturesCreator
from models.DAST.DASTModel import DAST_SentenceTrigger
from utils.VisdomHelper import EasyVisdom

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


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


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "mrpc":
        return acc_and_f1_and_p_and_r(preds, labels)
    else:
        raise KeyError(task_name)


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


def dataset_statistic(examples):
    """
    统计数据集的一些情况
    :param examples: 数据列表
    :return: 统计得到的字典
    """
    stat_result = dict()
    stat_result["data_number"] = len(examples)  # 数据数量
    stat_result["pos_example_number"] = len([ex for ex in examples if int(ex.label)==1])  # 正例数量
    stat_result["pos_rate"] = 0 if stat_result["data_number"]==0 else stat_result["pos_example_number"] / stat_result["data_number"]  # 正例比例
    stat_result["neg_example_number"] = stat_result["data_number"] - stat_result["pos_example_number"]
    stat_result["neg_rate"] = 1 - stat_result["pos_rate"]
    return stat_result


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

def do_eval(task_name, args, eval_examples, eval_features, model, output_mode, num_labels, tr_loss, nb_tr_steps, global_step, device):
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)  # 按顺序依次采样
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    likelihood = preds.copy()
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, all_label_ids.numpy())
    loss = None if not args.do_train or nb_tr_steps==0 else tr_loss / nb_tr_steps

    result['eval_loss'] = eval_loss
    result['global_step'] = global_step
    result['loss'] = loss
    # coref result
    coref_eval_result = evaluate_coref_from_example(eval_examples, preds)
    result = dict(result, **coref_eval_result)
    return result, preds, likelihood


def do_eval_cluster(task_name, eval_examples, preds, return_df=False):
    true_example_label = np.array([int(e.label) for e in eval_examples])
    pred_frame = examples_to_predict_frame(eval_examples, preds)
    cluster_frame = connected_components_clustering(pred_frame)
    cluster_preds = cluster_frame["# pred label"].astype(int).to_numpy()
    cluster_result = compute_metrics(task_name, cluster_preds, true_example_label)
    if return_df:
        return cluster_result, cluster_preds, cluster_frame
    else:
        return cluster_result, cluster_preds


def create_model_input_loader(feature_list: list, batch_size):
    all_BERT_input_ids = torch.tensor([f.BERT_input_ids for f in feature_list], dtype=torch.long)
    all_BERT_input_mask = torch.tensor([f.BERT_input_mask for f in feature_list], dtype=torch.long)
    all_BERT_segment_ids = torch.tensor([f.BERT_segment_ids for f in feature_list], dtype=torch.long)
    all_BERT_trigger_mask_a = torch.tensor([f.BERT_trigger_mask_a for f in feature_list], dtype=torch.long)
    all_BERT_trigger_mask_b = torch.tensor([f.BERT_trigger_mask_b for f in feature_list], dtype=torch.long)
    all_label = torch.tensor([f.label for f in feature_list], dtype=torch.long)
    BERT_data = TensorDataset(
        all_BERT_input_ids, all_BERT_input_mask, all_BERT_segment_ids,
        all_BERT_trigger_mask_a, all_BERT_trigger_mask_b, all_label)
    BERT_dataloader = DataLoader(BERT_data, batch_size=batch_size, shuffle=False)  # 之前已经手动shuffle
    return BERT_dataloader


def do_train(model, optimizer, device, BERT_dataloader, feature_list: list, n_gpu: int, fp16: bool,
             visdom_helper: EasyVisdom, num_train_optimization_steps: int, warmup_proportion: float,
             learning_rate: float, loss_each_step: dict, learning_rate_each_step: dict):
    if model is not None:
        model.train()
    global train_global_step
    train_BERT_dataloader = BERT_dataloader
    for step, batch in enumerate(tqdm(train_BERT_dataloader, desc="Train: ")):
        train_global_step += 1
        batch = tuple(t.to(device) for t in batch)
        BERT_input_ids, BERT_input_mask, BERT_segment_ids, BERT_trigger_mask_a, BERT_trigger_mask_b, labels = batch
        logits = model(BERT_input_ids, BERT_segment_ids, BERT_input_mask, BERT_trigger_mask_a, BERT_trigger_mask_b)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        # 学习率曲线
        lr_this_step = learning_rate * warmup_linear(train_global_step / num_train_optimization_steps, warmup_proportion)
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
            train_loss=loss.detach().cpu().item(), learning_rate=learning_rate)


def do_predict(model, device, BERT_dataloader, feature_list: list):
    if model is not None:
        model.eval()
    eval_BERT_dataloader = BERT_dataloader

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for step, batch in enumerate(tqdm(eval_BERT_dataloader, desc="Predict: ")):
        batch = tuple(t.to(device) for t in batch)
        BERT_input_ids, BERT_input_mask, BERT_segment_ids, BERT_trigger_mask_a, BERT_trigger_mask_b, labels = batch
        with torch.no_grad():
            logits = model(BERT_input_ids, BERT_segment_ids, BERT_input_mask, BERT_trigger_mask_a, BERT_trigger_mask_b)

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

        optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
    return optimizer


def compare_with_checkpoint(model, output_checkpoint_dir, logger, compare_dict: dict, eval_result: dict,
                            epoch: int, global_step: int, by_what: str="CoNLL_f1"):
    if eval_result[by_what] > compare_dict["best_result"]:
        logger.info(f"********** best model with {by_what}={eval_result[by_what]}, update checkpoint **********")
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
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
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
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

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
    feature_creator = InputFeaturesCreator(ECBPlus=EcbPlusTopView())
    logger.info("*******  Create eval features  ******")
    dev_features = feature_creator.create_from_dataset(
        topics=CONFIG.topics_test, cross_document=cross_document, cross_topic=cross_topic, ignore_order=True,
        max_seq_length=args.max_seq_length, trigger_half_window=args.trigger_half_window)

    num_train_optimization_steps = 0
    train_features = []
    if args.do_train:
        logger.info("*******  Create train features  ******")
        train_features = feature_creator.create_from_dataset(
            topics=CONFIG.topics_train, cross_document=cross_document, cross_topic=cross_topic, ignore_order=True,
            max_seq_length=args.max_seq_length, trigger_half_window=args.trigger_half_window)
        num_train_optimization_steps = int(len(train_features) / args.batch_size) * args.num_train_epochs
        if args.increase_positive >= 1:  # 扩增正例
            train_features = dataset_increase(examples=train_features, more=args.increase_positive, label=1)
        train_features = dataset_shuffle(train_features)

    # 准备模型
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_-1')
    if args.use_sentence_trigger_feature and not args.use_document_feature and not args.use_argument_feature:
        ModelClass = DAST_SentenceTrigger
    else:
        ModelClass = None

    if ModelClass is not None and not args.load_trained:
        model = ModelClass.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=2)
    elif ModelClass is not None and args.load_trained:
        model = load_model_from_checkpoint(load_checkpoint_dir=args.load_model_dir, num_labels=2, ModelClass=ModelClass)
    else:
        model = None
        raise ModuleNotFoundError("Please choose a model")

    if args.fp16:
        model.half()
    model.to(device)  # todo
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 准备绘图
    time_stamp = get_time_stamp()
    visdom_helper = EasyVisdom(env_name=f"DAST env: %s %s " % (time_stamp, args.description), enable=args.draw_visdom)

    dev_BERT_dataloader = create_model_input_loader(feature_list=dev_features, batch_size=args.batch_size)
    visdom_helper.show_dict(title_name="Command Line Arguments", dic=vars(args))
    visdom_helper.show_dict(title_name="Eval Data Statistics", dic=input_feature_statistics(dev_features))
    if args.do_train:
        # 准备优化器
        optimizer = prepare_optimizer(
            model, fp16=args.fp16, loss_scale=args.loss_scale, learning_rate=args.learning_rate,
            warmup_proportion=args.warmup_proportion, num_train_optimization_steps=num_train_optimization_steps)
        # train数据
        train_BERT_dataloader = create_model_input_loader(feature_list=train_features, batch_size=args.batch_size)
        visdom_helper.show_dict(title_name="Train Data Statistics", dic=input_feature_statistics(train_features))


        global train_global_step
        compare_dict = {"best_result": -1, "best_epoch": -1, "best_global_step": -1}
        train_curve_datas = {
            "train_loss_each_step": dict(),
            "learning_rate_each_step": dict(),
            "eval_result_each_epoch_on_train": dict(),
            "eval_result_each_epoch_on_dev": dict(),
        }

        # 训练开始前先eval一次
        train_result, train_pred = do_predict(
            model=model, device=device, BERT_dataloader=train_BERT_dataloader, feature_list=train_features)
        eval_result, eval_pred = do_predict(
            model=model, device=device, BERT_dataloader=dev_BERT_dataloader, feature_list=dev_features)
        train_curve_datas["eval_result_each_epoch_on_train"][0] = train_result
        train_curve_datas["eval_result_each_epoch_on_dev"][0] = eval_result
        draw_visdom_each_epoch(visdom_helper=visdom_helper, epoch=0, train_result=train_result, eval_result=eval_result)
        # 训练
        for epoch in trange(1, int(args.num_train_epochs)+1, desc="Epoch"):
            do_train(
                model=model, optimizer=optimizer, device=device, BERT_dataloader=train_BERT_dataloader,
                feature_list=train_features, n_gpu=n_gpu, fp16=args.fp16, visdom_helper=visdom_helper,
                num_train_optimization_steps=num_train_optimization_steps, warmup_proportion=args.warmup_proportion,
                learning_rate=args.learning_rate, loss_each_step=train_curve_datas["train_loss_each_step"],
                learning_rate_each_step=train_curve_datas["learning_rate_each_step"])
            train_result, train_pred = do_predict(
                model=model, device=device, BERT_dataloader=train_BERT_dataloader, feature_list=train_features)
            eval_result, eval_pred = do_predict(
                model=model, device=device, BERT_dataloader=dev_BERT_dataloader, feature_list=dev_features)
            train_curve_datas["eval_result_each_epoch_on_train"][epoch] = train_result
            train_curve_datas["eval_result_each_epoch_on_dev"][epoch] = eval_result
            compare_with_checkpoint(
                model=model, output_checkpoint_dir=args.train_output_dir, logger=logger, compare_dict=compare_dict,
                eval_result=eval_result, epoch=epoch, global_step=train_global_step, by_what=CONFIG.CHECKPOINT_BY_WHAT)
            draw_visdom_each_epoch(visdom_helper=visdom_helper, epoch=epoch, train_result=train_result, eval_result=eval_result)
        # 保存训练曲线的所有数据点
        best_eval_result =  train_curve_datas["eval_result_each_epoch_on_dev"][compare_dict["best_epoch"]]
        with open(os.path.join(args.train_output_dir, CONFIG.TRAIN_CURVE_DATA_FILE_NAME), "w") as f:
            json.dump(train_curve_datas, f)
        with open(os.path.join(args.train_output_dir, CONFIG.TRAIN_BEST_EVAL_RESULT_FILE_NAME), "w") as f:
            json.dump(best_eval_result, f, indent=4)
        # 展示最好结果
        visdom_helper.show_dict(title_name="Best Eval Result", dic=best_eval_result)
        # load最好的模型
        model = load_model_from_checkpoint(load_checkpoint_dir=args.train_output_dir, num_labels=2, ModelClass=ModelClass)

    if args.do_predict_only:
        eval_result, eval_pred = do_predict(model=model, device=device, BERT_dataloader=dev_BERT_dataloader, feature_list=dev_features)
        visdom_helper.show_dict(title_name="Eval Result", dic=eval_result)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
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
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
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
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
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
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--auto_choose_gpu",
                        action='store_true',
                        help="Whether to automatically choose GPU.")
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
    parser.add_argument("--eval_data_dir",
                        default="",
                        type=str,
                        help="The input data dir for evaluate. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_data_dir",
                        default="",
                        type=str,
                        help="The input data dir for train. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--increase_positive",
                        default=0,
                        type=int,
                        help="How many times positive examples should be increased.")
    parser.add_argument("--draw_visdom",
                        action="store_true",
                        help="Weather to use visdom to draw.")
    parser.add_argument("--do_cluster",
                        action="store_true",
                        help="Weather to do graph connection component cluster.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "mrpc": MrpcProcessor,  # 判断两个句子是否语义等价。和我的coreference任务最接近
    }

    output_modes = {
        "mrpc": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if os.path.exists(args.train_output_dir) and os.listdir(args.train_output_dir) and args.do_train:
        raise ValueError("Train output directory ({}) already exists and is not empty.".format(args.train_output_dir))
    if args.do_train and not os.path.exists(args.train_output_dir):
        os.makedirs(args.train_output_dir)

    if os.path.exists(args.predict_output_dir) and os.listdir(args.predict_output_dir) and args.do_eval:
        raise ValueError("Predict output directory ({}) already exists and is not empty.".format(args.train_output_dir))
    if args.do_predict and not os.path.exists(args.predict_output_dir):
        os.makedirs(args.predict_output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.train_data_dir)
        if args.increase_positive >= 1:  # 扩增正例
            train_examples = dataset_increase(examples=train_examples, more=args.increase_positive, label=1)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    # 测试数据
    eval_examples = processor.get_dev_examples(args.eval_data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)

    time_stamp = get_time_stamp()
    visdom_helper = VisdomHelper(env=f"train BERT Sentence Matching: {time_stamp} {args.description}", enable=args.draw_visdom)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        # visdom展示此次训练的命令行参数和训练数据统计
        train_examples_stat = dataset_statistic(examples=train_examples)
        visdom_helper.show_dict(text_name="args", dic=vars(args))
        visdom_helper.show_dict(text_name="train dataset statistics", dic=train_examples_stat)
        best_dict = {  # 记录什么时候达到最好结果
            "best_dev_epoch": 0,
            "best_dev_acc": -1.,

            "best_train_epoch": 0,
            "best_train_acc": -1.,

            "best_dev_epoch_cluster": 0,
            "best_dev_acc_cluster": -1.,

            "best_train_epoch_cluster": 0,
            "best_train_acc_cluster": -1.,

            "best_dev_BLANC_f1_epoch": 0,
            "best_dev_BLANC_f1": -1,
        }
        # 开始训练前先画一个acc点
        train_round = -1
        epoch = -1
        train_result, train_preds, train_likelihood = do_eval(
            task_name, args, train_examples, train_features, model, output_mode, num_labels, tr_loss,
            nb_tr_steps,
            global_step, device)
        visdom_helper.update_line(line_name="train_acc_epoch", round=epoch, value=train_result["acc"])
        visdom_helper.update_line(line_name="train_f1_epoch", round=epoch, value=train_result["f1"])
        visdom_helper.update_line(line_name="loss_on_train_set_epoch", round=epoch, value=train_result["eval_loss"])
        visdom_helper.update_line(line_name="train_B-cubed_f1_epoch", round=epoch, value=train_result["B-cubed_f1"])
        visdom_helper.update_line(line_name="train_MUC_f1_epoch", round=epoch, value=train_result["MUC_f1"])
        visdom_helper.update_line(line_name="train_CEAFe_f1_epoch", round=epoch, value=train_result["CEAFe_f1"])
        visdom_helper.update_line(line_name="train_CEAFm_f1_epoch", round=epoch, value=train_result["CEAFm_f1"])
        visdom_helper.update_line(line_name="train_BLANCc_f1_epoch", round=epoch, value=train_result["BLANCc_f1"])
        visdom_helper.update_line(line_name="train_BLANCn_f1_epoch", round=epoch, value=train_result["BLANCn_f1"])
        visdom_helper.update_line(line_name="train_BLANC_f1_epoch", round=epoch, value=train_result["BLANC_f1"])
        visdom_helper.update_line(line_name="train_CoNLL_f1_epoch", round=epoch, value=train_result["CoNLL_f1"])
        visdom_helper.update_line(line_name="train_AVG_f1_epoch", round=epoch, value=train_result["AVG_f1"])


        # visdom_helper.update_line(line_name="train_acc_step", round=train_round, value=train_result["acc"])
        # visdom_helper.update_line(line_name="train_f1_step", round=train_round, value=train_result["f1"])
        # visdom_helper.update_line(line_name="loss_on_train_set_step", round=train_round, value=train_result["eval_loss"])
        # visdom_helper.update_line(line_name="train_B-cubed_f1_step", round=train_round, value=train_result["B-cubed_f1"])
        # visdom_helper.update_line(line_name="train_MUC_f1_step", round=train_round, value=train_result["MUC_f1"])
        # visdom_helper.update_line(line_name="train_CEAFe_f1_step", round=train_round, value=train_result["CEAFe_f1"])
        # visdom_helper.update_line(line_name="train_CEAFm_f1_step", round=train_round, value=train_result["CEAFm_f1"])
        # visdom_helper.update_line(line_name="train_BLANC_f1_step", round=train_round, value=train_result["BLANC_f1"])

        # dev集
        dev_result, dev_preds, dev_likelihood = do_eval(
            task_name, args, eval_examples, eval_features, model, output_mode, num_labels, tr_loss,
            nb_tr_steps,
            global_step, device)
        visdom_helper.update_line(line_name="dev_acc_epoch", round=epoch, value=dev_result["acc"])
        visdom_helper.update_line(line_name="dev_f1_epoch", round=epoch, value=dev_result["f1"])
        visdom_helper.update_line(line_name="loss_on_dev_set_epoch", round=epoch, value=dev_result["eval_loss"])
        visdom_helper.update_line(line_name="dev_B-cubed_f1_epoch", round=epoch, value=dev_result["B-cubed_f1"])
        visdom_helper.update_line(line_name="dev_MUC_f1_epoch", round=epoch, value=dev_result["MUC_f1"])
        visdom_helper.update_line(line_name="dev_CEAFe_f1_epoch", round=epoch, value=dev_result["CEAFe_f1"])
        visdom_helper.update_line(line_name="dev_CEAFm_f1_epoch", round=epoch, value=dev_result["CEAFm_f1"])
        visdom_helper.update_line(line_name="dev_BLANCc_f1_epoch", round=epoch, value=dev_result["BLANCc_f1"])
        visdom_helper.update_line(line_name="dev_BLANCn_f1_epoch", round=epoch, value=dev_result["BLANCn_f1"])
        visdom_helper.update_line(line_name="dev_BLANC_f1_epoch", round=epoch, value=dev_result["BLANC_f1"])
        visdom_helper.update_line(line_name="dev_CoNLL_f1_epoch", round=epoch, value=dev_result["CoNLL_f1"])
        visdom_helper.update_line(line_name="dev_AVG_f1_epoch", round=epoch, value=dev_result["AVG_f1"])

        # visdom_helper.update_line(line_name="dev_acc_step", round=train_round, value=dev_result["acc"])
        # visdom_helper.update_line(line_name="dev_f1_step", round=train_round, value=dev_result["f1"])
        # visdom_helper.update_line(line_name="loss_on_dev_set_step", round=train_round, value=dev_result["eval_loss"])
        # visdom_helper.update_line(line_name="dev_B-cubed_f1_step", round=train_round, value=dev_result["B-cubed_f1"])
        # visdom_helper.update_line(line_name="dev_MUC_f1_step", round=train_round, value=dev_result["MUC_f1"])
        # visdom_helper.update_line(line_name="dev_CEAFe_f1_step", round=train_round, value=dev_result["CEAFe_f1"])
        # visdom_helper.update_line(line_name="dev_CEAFm_f1_step", round=train_round, value=dev_result["CEAFm_f1"])
        # visdom_helper.update_line(line_name="dev_BLANC_f1_step", round=train_round, value=dev_result["BLANC_f1"])

        train_round = -1
        epoch = -1
        train_evals = dict()
        dev_evals = dict()
        train_cluster_evals = dict()
        dev_cluster_evals = dict()
        draw_acc_each_epoch = False
        draw_acc_steps = num_train_optimization_steps / 100
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                train_round += 1
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # train loss图
                # 每个batch画一个
                visdom_helper.update_line(line_name="train_loss", round=train_round, value=loss.item())
                visdom_helper.update_scatter(line_name="train_loss_scatter", round=train_round, value=loss.item())
                # 每n个batch画一个
                visdom_helper.update_line_average(line_name="train_loss_average-6", round=train_round,
                                                  value=loss.item(), buf_size=6)
                visdom_helper.update_line_average(line_name="train_loss_average-5", round=train_round,
                                                  value=loss.item(), buf_size=5)
                visdom_helper.update_line_average(line_name="train_loss_average-10", round=train_round,
                                                  value=loss.item(), buf_size=10)
                visdom_helper.update_line_average(line_name="train_loss_average-11", round=train_round,
                                                  value=loss.item(), buf_size=11)
                visdom_helper.update_line_average(line_name="train_loss_average-50", round=train_round,
                                                  value=loss.item(), buf_size=50)
                visdom_helper.update_line_average(line_name="train_loss_average-100", round=train_round,
                                                  value=loss.item(), buf_size=100)
                # 每n个点做平滑
                visdom_helper.update_line_smooth(line_name="train_loss_smooth-3", round=train_round,
                                                 value=loss.item(), buf_size=3)
                visdom_helper.update_line_smooth(line_name="train_loss_smooth-6", round=train_round,
                                                 value=loss.item(), buf_size=6)
                visdom_helper.update_line_smooth(line_name="train_loss_smooth-5", round=train_round,
                                                 value=loss.item(), buf_size=5)
                visdom_helper.update_line_smooth(line_name="train_loss_smooth-10", round=train_round,
                                                 value=loss.item(), buf_size=10)
                visdom_helper.update_line_smooth(line_name="train_loss_smooth-11", round=train_round,
                                                 value=loss.item(), buf_size=11)
                visdom_helper.update_line_smooth(line_name="train_loss_smooth-50", round=train_round,
                                                 value=loss.item(), buf_size=50)
                visdom_helper.update_line_smooth(line_name="train_loss_smooth-100", round=train_round,
                                                 value=loss.item(), buf_size=100)
                visdom_helper.update_line_smooth(line_name="train_loss_smooth-3", round=train_round,
                                                 value=loss.item(), buf_size=3)
                # 总平均loss变化
                visdom_helper.update_line_total_average(line_name="train_loss_total_average", round=train_round,
                                                        value=loss.item())
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    current_lr = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)  # 当前的学习率(BertAdam)里会自动完成这步
                    visdom_helper.update_line(line_name="learning rate", round=train_round, value=current_lr)

                    # 训练中画acc曲线
                    # if not draw_acc_each_epoch and train_round % draw_acc_steps==0 or step == len(train_dataloader)-1:
                    #     train_result, train_preds, train_likelihood = do_eval(
                    #         task_name, args, train_examples, train_features, model, output_mode, num_labels, tr_loss,
                    #         nb_tr_steps,
                    #         global_step, device)
                    #     visdom_helper.update_line(line_name="train_acc_step", round=train_round, value=train_result["acc"])
                    #     visdom_helper.update_line(line_name="train_f1_step", round=train_round, value=train_result["f1"])
                    #     visdom_helper.update_line(line_name="loss_on_train_set_step", round=train_round,
                    #                               value=train_result["eval_loss"])
                    #     visdom_helper.update_line(line_name="train_B-cubed_f1_step", round=train_round,
                    #                               value=train_result["B-cubed_f1"])
                    #     visdom_helper.update_line(line_name="train_MUC_f1_step", round=train_round,
                    #                               value=train_result["MUC_f1"])
                    #     visdom_helper.update_line(line_name="train_CEAFe_f1_step", round=train_round,
                    #                               value=train_result["CEAFe_f1"])
                    #     visdom_helper.update_line(line_name="train_CEAFm_f1_step", round=train_round,
                    #                               value=train_result["CEAFm_f1"])
                    #     visdom_helper.update_line(line_name="train_BLANC_f1_step", round=train_round,
                    #                               value=train_result["BLANC_f1"])
                    #
                    #     # dev集
                    #     dev_result, dev_preds, dev_likelihood = do_eval(
                    #         task_name, args, eval_examples, eval_features, model, output_mode, num_labels, tr_loss,
                    #         nb_tr_steps,
                    #         global_step, device)
                    #     visdom_helper.update_line(line_name="dev_acc_step", round=train_round, value=dev_result["acc"])
                    #     visdom_helper.update_line(line_name="dev_f1_step", round=train_round, value=dev_result["f1"])
                    #     visdom_helper.update_line(line_name="loss_on_dev_set_step", round=train_round,
                    #                               value=dev_result["eval_loss"])
                    #     visdom_helper.update_line(line_name="dev_B-cubed_f1_step", round=train_round,
                    #                               value=dev_result["B-cubed_f1"])
                    #     visdom_helper.update_line(line_name="dev_MUC_f1_step", round=train_round,
                    #                               value=dev_result["MUC_f1"])
                    #     visdom_helper.update_line(line_name="dev_CEAFe_f1_step", round=train_round,
                    #                               value=dev_result["CEAFe_f1"])
                    #     visdom_helper.update_line(line_name="dev_CEAFm_f1_step", round=train_round,
                    #                               value=dev_result["CEAFm_f1"])
                    #     visdom_helper.update_line(line_name="dev_BLANC_f1_step", round=train_round,
                    #                               value=dev_result["BLANC_f1"])
                    #
                    #     # 为防止因为总是测试导致训练过慢，当dev acc高于一个阈值后就只每个epoch画一个
                    #     if dev_result["acc"] >= CONFIG.DRAW_ACC_THRESHOLD:
                    #         draw_acc_each_epoch = True

            # 评估模型, 画在train集和dev集上的eval曲线图
            # train集
            train_result, train_preds, train_likelihood = do_eval(
                task_name, args, train_examples, train_features, model, output_mode, num_labels, tr_loss, nb_tr_steps,
                global_step, device)
            train_cluster_result, train_cluster_preds = do_eval_cluster(task_name, train_examples, train_preds)
            visdom_helper.update_line(line_name="train_acc_epoch", round=epoch, value=train_result["acc"])
            visdom_helper.update_line(line_name="train_f1_epoch", round=epoch, value=train_result["f1"])
            visdom_helper.update_line(line_name="loss_on_train_set_epoch", round=epoch, value=train_result["eval_loss"])
            visdom_helper.update_line(line_name="train_acc_cluster_epoch", round=epoch, value=train_cluster_result["acc"])
            visdom_helper.update_line(line_name="train_f1_cluster_epoch", round=epoch, value=train_cluster_result["f1"])
            visdom_helper.update_line(line_name="train_B-cubed_f1_epoch", round=epoch, value=train_result["B-cubed_f1"])
            visdom_helper.update_line(line_name="train_MUC_f1_epoch", round=epoch, value=train_result["MUC_f1"])
            visdom_helper.update_line(line_name="train_CEAFe_f1_epoch", round=epoch, value=train_result["CEAFe_f1"])
            visdom_helper.update_line(line_name="train_CEAFm_f1_epoch", round=epoch, value=train_result["CEAFm_f1"])
            visdom_helper.update_line(line_name="train_BLANCc_f1_epoch", round=epoch, value=train_result["BLANCc_f1"])
            visdom_helper.update_line(line_name="train_BLANCn_f1_epoch", round=epoch, value=train_result["BLANCn_f1"])
            visdom_helper.update_line(line_name="train_BLANC_f1_epoch", round=epoch, value=train_result["BLANC_f1"])
            visdom_helper.update_line(line_name="train_CoNLL_f1_epoch", round=epoch, value=train_result["CoNLL_f1"])
            visdom_helper.update_line(line_name="train_AVG_f1_epoch", round=epoch, value=train_result["AVG_f1"])

            train_evals[epoch] = train_result
            train_cluster_evals[epoch] = train_cluster_result
            # dev集
            dev_result, dev_preds, dev_likelihood = do_eval(
                task_name, args, eval_examples, eval_features, model, output_mode, num_labels, tr_loss, nb_tr_steps,
                global_step, device)
            dev_cluster_result, dev_cluster_preds = do_eval_cluster(task_name, eval_examples, dev_preds)
            visdom_helper.update_line(line_name="dev_acc_epoch", round=epoch, value=dev_result["acc"])
            visdom_helper.update_line(line_name="dev_f1_epoch", round=epoch, value=dev_result["f1"])
            visdom_helper.update_line(line_name="loss_on_dev_set_epoch", round=epoch, value=dev_result["eval_loss"])
            visdom_helper.update_line(line_name="dev_acc_cluster_epoch", round=epoch, value=dev_cluster_result["acc"])
            visdom_helper.update_line(line_name="dev_f1_cluster_epoch", round=epoch, value=dev_cluster_result["f1"])
            visdom_helper.update_line(line_name="dev_B-cubed_f1_epoch", round=epoch, value=dev_result["B-cubed_f1"])
            visdom_helper.update_line(line_name="dev_MUC_f1_epoch", round=epoch, value=dev_result["MUC_f1"])
            visdom_helper.update_line(line_name="dev_CEAFe_f1_epoch", round=epoch, value=dev_result["CEAFe_f1"])
            visdom_helper.update_line(line_name="dev_CEAFm_f1_epoch", round=epoch, value=dev_result["CEAFm_f1"])
            visdom_helper.update_line(line_name="dev_BLANCc_f1_epoch", round=epoch, value=dev_result["BLANCc_f1"])
            visdom_helper.update_line(line_name="dev_BLANCn_f1_epoch", round=epoch, value=dev_result["BLANCn_f1"])
            visdom_helper.update_line(line_name="dev_BLANC_f1_epoch", round=epoch, value=dev_result["BLANC_f1"])
            visdom_helper.update_line(line_name="dev_CoNLL_f1_epoch", round=epoch, value=dev_result["CoNLL_f1"])
            visdom_helper.update_line(line_name="dev_AVG_f1_epoch", round=epoch, value=dev_result["AVG_f1"])

            dev_evals[epoch] = dev_result
            dev_cluster_evals[epoch] = dev_cluster_result

            # 把目前为止在dev上acc最好的模型保存为checkpoint
            if train_result["acc"] > best_dict["best_train_acc"]:
                best_dict["best_train_acc"] = train_result["acc"]
                best_dict["best_train_epoch"] = epoch
            if train_cluster_result["acc"] > best_dict["best_train_acc_cluster"]:
                best_dict["best_train_acc_cluster"] = train_cluster_result["acc"]
                best_dict["best_train_epoch_cluster"] = epoch
            if dev_cluster_result["acc"] > best_dict["best_dev_acc_cluster"]:
                best_dict["best_dev_acc_cluster"] = dev_cluster_result["acc"]
                best_dict["best_dev_epoch_cluster"] = epoch
            # 依据dev BLANC f1保存checkpoint
            if dev_result["BLANC_f1"] > best_dict["best_dev_BLANC_f1"]:
                best_dict["best_dev_BLANC_f1"] = dev_result["BLANC_f1"]
                best_dict["best_dev_BLANC_f1_epoch"] = epoch
                store_model_to_checkpoint(model, output_checkpoint_dir=args.train_output_dir)
                # 分类评测
                best_dev_eval_file = os.path.join(args.train_output_dir, "eval_results.txt")
                with open(best_dev_eval_file, "w") as writer:
                    logger.info("*****Best Dev Eval Results, save checkpoint *****")
                    for key in sorted(dev_result.keys()):
                        logger.info("  %s = %s", key, str(dev_result[key]))
                        writer.write("%s = %s\n" % (key, str(dev_result[key])))
                    for key in sorted(dev_cluster_result.keys()):
                        logger.info("  cluster %s = %s", key, str(dev_cluster_result[key]))
                        writer.write("cluster %s = %s\n" % (key, str(dev_cluster_result[key])))

        # Load a trained model and config that you have fine-tuned
        model = load_model_from_checkpoint(load_checkpoint_dir=args.train_output_dir, num_labels=num_labels)

        visdom_helper.show_dict(text_name="training_log", dic=best_dict)
        visdom_helper.show_dict(text_name="best_dev_eval_result", dic=dev_evals[best_dict["best_dev_BLANC_f1_epoch"]])
        visdom_helper.show_dict(text_name="best_dev_eval_result_cluster", dic=dev_cluster_evals[best_dict["best_dev_epoch_cluster"]])
        # 实验记录
        generate_record(
            table_name="train_BERT",
            coref_level=args.coref_level,
            input_data_path=os.path.join(args.train_data_dir, "train.tsv"),
            model_parameter_path=args.train_output_dir,
            args_dict=vars(args),
            time_stamp=time_stamp)

    elif args.load_trained:
        # Load a trained model and config that you have fine-tuned
        model = load_model_from_checkpoint(load_checkpoint_dir=args.load_model_dir, num_labels=num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # logger.info("***** Running evaluation *****")
        # logger.info("  Num examples = %d", len(eval_examples))
        # logger.info("  Batch size = %d", args.eval_batch_size)
        # all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        # all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        #
        # if output_mode == "classification":
        #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        # elif output_mode == "regression":
        #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
        #
        # eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # # Run prediction for full data
        # eval_sampler = SequentialSampler(eval_data)  # 按顺序依次采样
        # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        #
        # model.eval()
        # eval_loss = 0
        # nb_eval_steps = 0
        # preds = []
        #
        # for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        #     input_ids = input_ids.to(device)
        #     input_mask = input_mask.to(device)
        #     segment_ids = segment_ids.to(device)
        #     label_ids = label_ids.to(device)
        #
        #     with torch.no_grad():
        #         logits = model(input_ids, segment_ids, input_mask, labels=None)
        #
        #     # create eval loss and other metric required by the task
        #     if output_mode == "classification":
        #         loss_fct = CrossEntropyLoss()
        #         tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        #     elif output_mode == "regression":
        #         loss_fct = MSELoss()
        #         tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        #
        #     eval_loss += tmp_eval_loss.mean().item()
        #     nb_eval_steps += 1
        #     if len(preds) == 0:
        #         preds.append(logits.detach().cpu().numpy())
        #     else:
        #         preds[0] = np.append(
        #             preds[0], logits.detach().cpu().numpy(), axis=0)
        #
        # eval_loss = eval_loss / nb_eval_steps
        # preds = preds[0]
        # likelihood = preds.copy()
        # if output_mode == "classification":
        #     preds = np.argmax(preds, axis=1)
        # elif output_mode == "regression":
        #     preds = np.squeeze(preds)
        # result = compute_metrics(task_name, preds, all_label_ids.numpy())
        # loss = tr_loss / nb_tr_steps if args.do_train else None
        #
        # result['eval_loss'] = eval_loss
        # result['global_step'] = global_step
        # result['loss'] = loss

        result, preds, likelihood = do_eval(task_name, args, eval_examples, eval_features, model, output_mode,
                                            num_labels, tr_loss, nb_tr_steps, global_step, device)
        result_cluster, preds_cluster, cluster_frame = do_eval_cluster(task_name, eval_examples, preds, return_df=True)
        # if args.do_train:
        #     train_eval_file = os.path.join(args.train_output_dir, "eval_results.txt")
        #     with open(train_eval_file, "w") as writer:
        #         logger.info("***** Eval results *****")
        #         for key in sorted(result.keys()):
        #             logger.info("  %s = %s", key, str(result[key]))
        #             writer.write("%s = %s\n" % (key, str(result[key])))
        # 绘制eval结果
        eval_examples_stat = dataset_statistic(eval_examples)
        visdom_helper.show_dict(text_name="eval_result", dic=result)
        visdom_helper.show_dict(text_name="eval_result_cluster", dic=result_cluster)
        visdom_helper.show_dict(text_name="eval dataset statistics", dic=eval_examples_stat)


        # 存储预测结果
        dev_data_csv = read_csv(os.path.join(args.eval_data_dir, "dev.tsv"), sep="\t")
        predict_result = dev_data_csv.loc[:, ["#1 ID", "#2 ID"]]
        predict_result["# 0 likelihood"] = Series(likelihood[:, 0])
        predict_result["# 1 likelihood"] = Series(likelihood[:, 1])
        predict_result["# pred label"] = Series(preds)
        output_pred_file = os.path.join(args.predict_output_dir, CONFIG.BERT_PREDICT_FILE_NAME)
        if not os.path.exists(args.predict_output_dir):
            os.makedirs(args.predict_output_dir)
        predict_result.to_csv(output_pred_file, sep="\t", index=False)
        output_pred_file_cluster = os.path.join(args.predict_output_dir, "BERT_predict_cluster.tsv")
        cluster_frame.to_csv(output_pred_file_cluster, sep="\t", index=False)

        predict_eval_file = os.path.join(args.predict_output_dir, "eval_results.txt")
        with open(predict_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            for key in sorted(result_cluster.keys()):
                logger.info("  cluster %s = %s", key, str(result_cluster[key]))
                writer.write("cluster %s = %s\n" % (key, str(result_cluster[key])))


        # 实验记录
        model_parameter_path = args.train_output_dir if args.do_train else args.load_model_dir
        generate_record(
            table_name="test_BERT",
            coref_level=args.coref_level,
            input_data_path=os.path.join(args.eval_data_dir, "dev.tsv"),
            model_parameter_path= model_parameter_path,
            output_evaluate_dir=args.predict_output_dir,
            args_dict=vars(args),
            eval_result_dict=result,
            time_stamp=time_stamp
        )

        # hack for MNLI-MM
        if task_name == "mnli":
            task_name = "mnli-mm"
            processor = processors[task_name]()

            if os.path.exists(args.train_output_dir + '-MM') and os.listdir(args.train_output_dir + '-MM') and args.do_train:
                raise ValueError("Output directory ({}) already exists and is not empty.".format(args.train_output_dir))
            if not os.path.exists(args.train_output_dir + '-MM'):
                os.makedirs(args.train_output_dir + '-MM')

            eval_examples = processor.get_dev_examples(args.eval_data_dir)
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            preds = np.argmax(preds, axis=1)
            result = compute_metrics(task_name, preds, all_label_ids.numpy())
            loss = tr_loss / nb_tr_steps if args.do_train else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.train_output_dir + '-MM', "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    # main()
    my_main()