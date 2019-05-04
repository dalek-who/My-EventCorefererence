import os

CUDA_VISIBLE_DEVICES = "3"
script = "main.py"
log_name = "log_run_gnn.txt"


args_value = {
    # 会调整的
    "--description": "DAST_within-document_batch-500_epoch-10_argument_no-hidden-layer_increase-positive-20_lr-5e-5_conv-8",
    "--coref_level": "within_document",  # "cross_document", "within_document", "cross_topic"
    "--learning_rate": 5e-5,
    "--max_seq_length": 123,
    "--train_batch_size": 500,
    "--eval_batch_size": 500,
    "--num_train_epochs": 10,
    "--increase_positive": 20,
    "--trigger_half_window": 3,

    # 各种目录
    "--load_model_dir": "",
    "--train_output_dir":
        # "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/DAST_model/demo",
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/DAST_model/Argument_no-hidden-layer_within-document_increase-positive-20_epoch-10_lr-5e-5_half-window-3_conv-8",
    "--predict_output_dir":
        # "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/DAST_predict/demo",
        "/users/wangyuanzheng/event_coreference/my-ev-coref/middle_data/DAST_predict/Argument_no-hidden-layer_within-document_increase-positive-20_epoch-10_lr-5e-5_half-window-3_conv-8",

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
    "--use_sentence_trigger_feature": False,
    "--use_argument_feature": True,


    # 使用默认值
    "--do_lower_case": True,
    "--no_cuda": False,
    "--fp16": False,
}

cmd_args = " ".join(["%s %s" % (k,v) for k,v in args_value.items() if v!=""]) + " " \
           + " ".join([k for k,v in args_store_true.items() if v==True])

command = f"""CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python {script} {cmd_args} >{log_name} 2>&1 &"""
print(command)
os.system(command)
