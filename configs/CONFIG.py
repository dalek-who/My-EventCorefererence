import os

topics_train = list(range(1, 36))
topics_test = list(range(36, 46))
topics_validation = [2, 5, 12, 18, 21, 23, 34, 35]

BERT_PREDICT_FILE_NAME = "BERT_predict.tsv"
EXPERIMENT_RECOED_DIR = os.path.abspath(os.path.join(__file__, "../../ExperimentRecord/"))
