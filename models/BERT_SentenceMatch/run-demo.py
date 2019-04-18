# 用来进行一些小的测试，例如batch可以开到多大

import os

CUDA_VISIBLE_DEVICES = 0
script = "BERT_SentenceMatch.py"
log_name = "log_run-demo.txt"

args_value = {
    # 会调整的
    "--learning_rate": 5e-6,
    "--max_seq_length": 90,
    "--train_batch_size": 65,
    "--eval_batch_size": 40,
    "--num_train_epochs": 3.0,
    "--coref_level": "cross_document",  # 或者"within_document", "cross_topic"
    "--description": "this_is_a_toy",
    "--increase_positive": 0,
    # 各种目录
    "--load_model_dir": "",
    "--predict_output_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_predict/demo",
    "--train_output_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_model/demo",
    "--eval_data_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_input/demo",
    "--train_data_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_input/demo",

    # 基本不变的
    "--bert_model": "bert-base-uncased",
    "--task_name": "MRPC",

    # 使用默认值
    "--cache_dir": "",
    "--warmup_proportion": 0.1,
    "--local_rank": -1,
    "--seed": 42,
    "--gradient_accumulation_steps": 1,
    "--loss_scale": 0,
    "--server_ip": "",
    "--server_port": "",
}

args_store_true = {
    # 会调整的
    "--do_train": True,
    "--do_eval": True,
    "--do_predict": False,
    "--load_trained": False,
    "--draw_visdom": False,

    # 使用默认值
    "--do_lower_case": True,
    "--no_cuda": False,
    "--fp16": False,
    "--auto_choose_gpu": False,
}

cmd_args = " ".join(["%s %s" % (k,v) for k,v in args_value.items() if v!=""]) + " " \
           + " ".join([k for k,v in args_store_true.items() if v==True])

command = f"""CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python {script} {cmd_args} >{log_name} 2>&1 &"""
print(command)
os.system(command)
