import os

topics_train = list(range(1, 36))
topics_test = list(range(36, 46))
topics_validation = [2, 5, 12, 18, 21, 23, 34, 35]

# 文件名
BERT_PREDICT_FILE_NAME = "BERT_predict.tsv"
EXPERIMENT_RECOED_DIR = os.path.abspath(os.path.join(__file__, "../../ExperimentRecord/"))
TRAIN_CURVE_DATA_FILE_NAME = "train_curve_datas.json"

# ECB+数据集路径
ECB_DRI = os.path.abspath(os.path.join(__file__, "../../datasets/ECB+_LREC2014/"))
CSV_DIR =  os.path.join(ECB_DRI, "ECBplus_coreference_sentences.csv")
DATA_DIR = os.path.join(ECB_DRI, "ECB+/")

# 选择模型的指标
CHECKPOINT_BY_WHAT = "CoNLL_f1"

# 模型的一些超参数
VECTOR_EMBEDDING_DIM = 16  # GNN中每个点embedding为几维
