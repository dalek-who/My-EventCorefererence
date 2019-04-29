from __future__ import absolute_import, division, print_function

import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import argparse
from tqdm import tqdm, trange
from utils.VisdomHelper import EasyVisdom

from time import sleep

def do_train(visdom_helper: EasyVisdom):
    global train_global_step
    for step, batch in enumerate(tqdm(range(2000), desc="Train: ")):
        train_global_step += 1

        loss = step * 2
        lr_this_step = step * 0.1
        # 绘图
        draw_visdom_each_step(
            visdom_helper=visdom_helper, step=train_global_step,
            train_loss=loss, learning_rate=lr_this_step)
    sleep(0.1)


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


train_global_step = 0
def my_main():
    """
    定义main的命令行参数args
    :return: args
    """
    parser = argparse.ArgumentParser()
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


    train_stat = {"data_num": 20000, "pos_num":2000, "pos_rate": 0.1}
    # 准备绘图
    visdom_helper = EasyVisdom(env_name=f"test why can't show picture", enable=True)

    visdom_helper.show_dict(title_name="Command Line Arguments", dic=vars(args))
    visdom_helper.show_dict(title_name="Eval Data Statistics", dic=train_stat)
    if args.do_train:
        visdom_helper.show_dict(title_name="Train Data Statistics", dic=train_stat)

        compare_dict = {"best_result": -1, "best_epoch": -1, "best_global_step": -1}
        train_curve_datas = {
            "train_loss_each_step": dict(),
            "learning_rate_each_step": dict(),
            "eval_result_each_epoch_on_train": dict(),
            "eval_result_each_epoch_on_dev": dict(),
        }

        # 训练开始前先eval一次
        result = {
            "precision": 0.3076923076923077,
            "recall": 0.28623853211009176,
            "acc": 0.9311691935633895,
            "f1": 0.2965779467680608,
            "acc_and_f1": 0.6138735701657252,
            "B-cubed_p": 0.8407298965143207,
            "B-cubed_r": 0.875752445447705,
            "B-cubed_f1": 0.8578838766168921,
            "MUC_p": 0.4592391304347826,
            "MUC_r": 0.4482758620689655,
            "MUC_f1": 0.45369127516778524,
            "CEAFe_p": 0.8246079616806107,
            "CEAFe_r": 0.829928013046292,
            "CEAFe_f1": 0.8272594342262074,
            "CEAFm_p": 0.7703160270880361,
            "CEAFm_r": 0.7703160270880361,
            "CEAFm_f1": 0.7703160270880361,
            "BLANCc_p": 0.18440366972477065,
            "BLANCc_r": 0.3688073394495413,
            "BLANCc_f1": 0.2458715596330275,
            "BLANCn_p": 0.9997806144835257,
            "BLANCn_r": 0.9994332384905655,
            "BLANCn_f1": 0.9996068963076628,
            "BLANC_p": 0.5920921421041482,
            "BLANC_r": 0.6841202889700534,
            "BLANC_f1": 0.6227392279703451,
            "CoNLL_p": 0.708192329543238,
            "CoNLL_r": 0.7179854401876543,
            "CoNLL_f1": 0.7129448620036283,
            "AVG_p": 0.6791672826834655,
            "AVG_r": 0.709519152383254,
            "AVG_f1": 0.6903934534953075,
            "eval_loss": 0.2027062310112847,
        }
        train_result = result
        eval_result = {k: 1 - v for k, v in result.items()}
        draw_visdom_each_epoch(visdom_helper=visdom_helper, epoch=0, train_result=train_result, eval_result=eval_result)
        # 训练
        for epoch in trange(1, int(args.num_train_epochs)+1, desc="Epoch"):
            do_train(visdom_helper=visdom_helper)
            train_result = result
            eval_result = {k:1-v for k,v in result.items()}
            draw_visdom_each_epoch(visdom_helper=visdom_helper, epoch=epoch, train_result=train_result, eval_result=eval_result)
            sleep(3)
        # 展示最好结果
        visdom_helper.show_dict(title_name="Best Eval Result", dic=result)
    print("hellow world")


if __name__ == "__main__":
    # main()
    my_main()