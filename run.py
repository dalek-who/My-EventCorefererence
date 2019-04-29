import os

CUDA_VISIBLE_DEVICES = "1,2,3"
script = "main.py"
log_name = "log_run.txt"


args_value = {
    # 会调整的
    "--description": "DAST_within-document_batch-120_epoch-3_sentence_and_trigger_classifier_no_hidden_layer_half_window_0",
    "--coref_level": "within_document",  # "cross_document", "within_document", "cross_topic"
    "--learning_rate": 5e-5,
    "--max_seq_length": 123,
    "--batch_size": 120,
    "--num_train_epochs": 4.0,
    "--increase_positive": 2,
    "--trigger_half_window": 0,

    # 各种目录
    "--load_model_dir": "",
    "--train_output_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/DAST_model/SentenceTrigger_half-window-3_no-hidden-layer_half-window-0",
    "--predict_output_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/DAST_predict/SentenceTrigger_half-window-3_no-hidden-layer_half-window-0",

    # 基本不变的
    "--bert_model": "bert-base-uncased",

    # 使用默认值
    "--cache_dir": "",
    "--warmup_proportion": 0.1,
    "--seed": 42,
    "--loss_scale": 0,
}

args_store_true = {
    # 会调整的
    "--do_train": True,
    "--do_predict_only": False,
    "--load_trained": False,
    "--draw_visdom": False,

    "--use_document_feature": False,
    "--use_sentence_trigger_feature": True,
    "--use_argument_feature": False,


    # 使用默认值
    "--do_lower_case": True,
    "--no_cuda": False,
    "--fp16": True,
}

cmd_args = " ".join(["%s %s" % (k,v) for k,v in args_value.items() if v!=""]) + " " \
           + " ".join([k for k,v in args_store_true.items() if v==True])

command = f"""CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python {script} {cmd_args} >{log_name} 2>&1 &"""
print(command)
os.system(command)
