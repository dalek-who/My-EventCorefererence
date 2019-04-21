import os

CUDA_VISIBLE_DEVICES = "0,1,2,3"
script = "BERT_SentenceMatch.py"
log_name = "log_run.txt"

args_value = {
    # 会调整的
    "--description": "cross_document,draw_CoNLL_and_AVG_each_epoch",
    "--coref_level": "cross_document",  # "cross_document", "within_document", "cross_topic"
    "--learning_rate": 5e-6,
    "--max_seq_length": 90,
    "--train_batch_size": 180,
    "--eval_batch_size": 180,
    "--num_train_epochs": 3.0,
    "--increase_positive": 0,
    # 各种目录
    "--load_model_dir": "",
    "--predict_output_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_predict/train-large_cdoc_wtpc_fix-draw_CoNLL",
    "--train_output_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_model/train-large_cdoc_wtpc_fix-draw_CoNLL",
    "--eval_data_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_input_fix/test-large_cdoc_wtpc",
    "--train_data_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/BERT_SentenceMatching_input_fix/train-large_cdoc_wtpc",

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
    "--do_predict": True,
    "--load_trained": False,
    "--draw_visdom": True,
    "--do_cluster": True,

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
