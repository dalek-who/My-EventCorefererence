import os

CUDA_VISIBLE_DEVICES = "1,2,3"
script = "main.py"
log_name = "log_run.txt"


args_value = {
    # 会调整的
    "--description": "DAST_cross-document_batch-123_epoch-3_sentence-and-trigger_no-hidden-layer_half-window-3_increase-positive-3",
    "--coref_level": "cross_document",  # "cross_document", "within_document", "cross_topic"
    "--learning_rate": 5e-5,
    "--max_seq_length": 123,
    "--batch_size": 180,
    "--num_train_epochs": 3.0,
    "--increase_positive": 3,
    "--trigger_half_window": 3,

    # 各种目录
    "--load_model_dir": "",
    "--train_output_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/DAST_model/SentenceTrigger_half-window-3_no-hidden-layer_cross-document_increase-positive-3",
    "--predict_output_dir":
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/DAST_predict/SentenceTrigger_half-window-3_no-hidden-layer_cross-document_increase-positive-3",

    # 基本不变的
    "--bert_model": "bert-base-uncased",
    "--cache_dir": "/users/wangyuanzheng/.pytorch_pretrained_bert",

    # 使用默认值
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
