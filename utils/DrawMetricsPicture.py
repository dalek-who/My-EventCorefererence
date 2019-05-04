import os
from collections import deque
from pandas import Series, DataFrame, Index
import matplotlib.pyplot as plt
import numpy as np
import json

# 平滑
def smooth_loss(loss_list, smooth_size: int):
    queue = deque(maxlen=smooth_size)
    smooth_loss_list = []
    for loss in loss_list:
        queue.append(loss)
        smooth_loss_list.append(np.average(queue))
    sr_smooth_loss = Series(smooth_loss_list, name="Train Loss")
    plot_smooth_loss = sr_smooth_loss.plot(title="Train Loss each %s step" % smooth_size, legend=True)
    plot_smooth_loss.set_xlabel("Step")
    plot_smooth_loss.set_ylabel("Loss")
    return plot_smooth_loss


def result_visualize(curve_data_dict: dict, output_dir):
    if not os.path.exists(os.path.join(output_dir, "./visualize")):
        os.makedirs(os.path.join(output_dir, "./visualize"))
    # train loss each step 曲线
    # 未平滑的
    sr_train_loss_each_step = Series(curve_data_dict['train_loss_each_step'], name="Train Loss")
    sr_train_loss_each_step.index = Index([int(i) for i in sr_train_loss_each_step.index])
    plot_train_loss_each_step = sr_train_loss_each_step.plot(title="Train Loss each step", legend=True)
    plot_train_loss_each_step.set_xlabel("Step")
    plot_train_loss_each_step.set_ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "./visualize/train_loss_each_step.PNG"))
    plt.clf()
    # 平滑的
    # 50 step平滑
    plot_train_loss_each_50_step = smooth_loss(sr_train_loss_each_step, smooth_size=50)
    plt.savefig(os.path.join(output_dir, "./visualize/train_loss_each_step_each_50_step.png"))
    plt.clf()
    # 100 step平滑
    plot_train_loss_each_100_step = smooth_loss(sr_train_loss_each_step, smooth_size=100)
    plt.savefig(os.path.join(output_dir, "./visualize/train_loss_each_step_each_100_step.png"))
    plt.clf()

    # learning rate each step 曲线
    sr_learning_rate_each_step = Series(curve_data_dict['learning_rate_each_step'], name="Learning Rate")
    sr_learning_rate_each_step.index = Index([int(i) for i in sr_learning_rate_each_step.index])
    plot_learning_rate_each_step = sr_learning_rate_each_step.plot(title="Learning Rate", legend=True)
    plot_learning_rate_each_step.set_xlabel("Step")
    plot_learning_rate_each_step.set_ylabel("Learning Rate")
    plt.savefig(os.path.join(output_dir, "./visualize/learning_rate_each_step.png"))
    plt.clf()

    # 测评指标曲线
    df_metrics = {
        "train": DataFrame(curve_data_dict["eval_result_each_epoch_on_train"]).T,
        "dev": DataFrame(curve_data_dict["eval_result_each_epoch_on_dev"]).T,
    }
    df_metrics["train"].index = Index([int(i) for i in df_metrics["train"].index])
    df_metrics["dev"].index = Index([int(i) for i in df_metrics["dev"].index])
    # 每个指标一张图, 图里有train和dev两条线
    for which_line in ("MUC_f1", "B-cubed_f1", "CEAFe_f1", "CEAFm_f1", "BLANCc_f1", "BLANCn_f1", "BLANC_f1",
                       "CoNLL_f1", "AVG_f1"):
        df = DataFrame()
        for which_set in ("train", "dev"):
            df[which_set + "_" + which_line] = df_metrics[which_set][which_line]
        metrics_plot = df.plot(title="%s of each epoch" % which_line)
        metrics_plot.set_xlabel("Epoch")
        metrics_plot.set_ylabel("Evaluation Result")
        plt.savefig(os.path.join(output_dir, "./visualize/train_and_dev_each_epoch_%s.png" % which_line))
        plt.clf()

    # 一张loss图，里面有train和dev两条线
    df = DataFrame()
    for which_set in ("train", "dev"):
        df[which_set + "_" + "eval_loss"] = df_metrics[which_set]["eval_loss"]
    metrics_plot = df.plot(title="Loss of each epoch")
    metrics_plot.set_xlabel("Epoch")
    metrics_plot.set_ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "./visualize/train_and_dev_each_epoch_Loss.png"))
    plt.clf()

    # 四张图，分别是train/dev的单项指标/复合指标
    for which_set in ("train", "dev"):
        for group_name, metrics_group in {"Single Metric": ["MUC_f1", "B-cubed_f1", "CEAFe_f1", "BLANC_f1"],
                                          "Compisite Metrics": ["CoNLL_f1", "AVG_f1"]}.items():
            metrics_plot = df_metrics[which_set][metrics_group].plot(title=group_name)
            metrics_plot.set_xlabel("Epoch")
            metrics_plot.set_ylabel("Evaluation Result")
            plt.savefig(os.path.join(output_dir, "./visualize/%s_%s_each_epoch.png" % (group_name, which_set)))
            plt.clf()

    # 论文中的评价指标表格
    metrics_use = "CoNLL_f1"  # 依据哪个指标选模型
    column_in_table = [
        'MUC_r', 'MUC_p', 'MUC_f1',
        'B-cubed_r', 'B-cubed_p', 'B-cubed_f1',
        'CEAFm_f1',
        'CEAFe_r', 'CEAFe_p', 'CEAFe_f1',
        'BLANC_r', 'BLANC_p', 'BLANC_f1',
        'CoNLL_f1',
        'AVG_f1',
    ]
    dev_metrics_in_table = df_metrics["dev"].loc[df_metrics["dev"][metrics_use] == max(df_metrics["dev"][metrics_use]),
                                                 column_in_table].head(1)
    dev_metrics_in_table.to_csv(os.path.join(output_dir, "./visualize/best_result.csv"), index=False)
