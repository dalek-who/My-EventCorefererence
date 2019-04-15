""" 提取特征"""
import sys
sys.path.append("../..")
import os
from pandas import DataFrame

from preprocessing.Structurize.EcbClass import *
from preprocessing.Feature.MentionPair import MentionPairCreator

topics_train = {
    "train-toy-2": [2],
    "test-toy-3": [3],

    "train-large": list(range(1, 36)),
    "train-middle-20": list(range(1, 21)),
    "train-middle-10": list(range(1, 11)),
    "train-small-5": list(range(1, 6)),
    "train-small-2": list(range(1, 3)),

    "test-large": list(range(36, 46)),
    "test-small": list(range(36,41)),
}

ECB = EcbPlusTopView()
MPC = MentionPairCreator(ECB)

to_csv = True
positive_increase = 0

def positive_rate(BERT_feature_table: DataFrame):
    pos_num = len(BERT_feature_table.loc[ BERT_feature_table.loc[:, "Quality"]==1, "Quality"])
    tot_num = len(BERT_feature_table.loc[:, "Quality"])
    return pos_num / tot_num

for k, topic_list in topics_train.items():
    for cross_document, cross_topic in [(False, False), (True, False), (True, True)]:
        csv_cross_doc = "_cdoc" if cross_document else "_wdoc"
        csv_cross_topic = "_ctpc" if cross_topic else "_wtpc"
        BERT_csv_dir = "./BERT_SentenceMatching/%s%s%s/" % (k, csv_cross_doc, csv_cross_topic)
        if not os.path.exists(BERT_csv_dir):
            os.makedirs(BERT_csv_dir)
        file_name = "train.tsv" if k.startswith("train") else "dev.tsv"
        BERT_csv_path = os.path.join(BERT_csv_dir, file_name)
        print(BERT_csv_path)
        table = MPC.BERT_MRPC_feature( topics=topic_list, cross_document=cross_document, cross_topic=cross_topic,
            positive_increase=positive_increase, shuffle=True, csv_path=BERT_csv_path, to_csv=to_csv)
        print("data number: %s,  positive_rate: %s" % (len(table.index), positive_rate(table)))
