from utils.ExperimentRecordManager import generate_record

# train:
rcd = generate_record(
    table_name="train_BERT",
    coref_level="within_document",
    input_data_path="../../datasets/MRPC/train.tsv",
    model_parameter_path="../../train_result_xxx",
    args_dict={"fp16": True, "epoch": 3, "learning_rate": 0.01},
    description="first fake test")
print(rcd)

# test:
rcd = generate_record(
    table_name="test_BERT",
    coref_level="global",
    input_data_path="../../datasets/MRPC/dev.tsv",
    model_parameter_path="../../train_MatchZoo.bin",
    output_evaluate_dir="../../predict_result",
    args_dict={"fp16": True, "epoch": 3, "learning_rate": 0.01},
    eval_result_dict={"p": 0.8, "f1": 0.4},
    description="second fake test")
print(rcd)
