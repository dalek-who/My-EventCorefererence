""" 提取特征"""
#%%
from __future__ import absolute_import, division, print_function
import sys
sys.path.append("../")
import os
from pandas import DataFrame

from preprocessing.Structurize.EcbClass import *
from preprocessing.Feature.MentionPair import MentionPairCreator

topics_train = {
    "train-large": list(range(1, 36)),
    "test-large": list(range(36, 46)),
}

ECB = EcbPlusTopView()
MPC = MentionPairCreator(ECB)

to_csv = True
topic_list = topics_train["test-large"]


#%%
for k, topic_list in topics_train.items():
    for cross_document, cross_topic in [(False, False), (True, False), (True, True)]:
        csv_cross_doc = "_cdoc" if cross_document else "_wdoc"
        csv_cross_topic = "_ctpc" if cross_topic else "_wtpc"
        BERT_csv_dir = "./BERT_SentenceMatching_input_fix/%s%s%s/" % (k, csv_cross_doc, csv_cross_topic)
        if not os.path.exists(BERT_csv_dir):
            os.makedirs(BERT_csv_dir)
        file_name = "train.tsv" if k.startswith("train") else "dev.tsv"
        BERT_csv_path = os.path.join(BERT_csv_dir, file_name)
        print(BERT_csv_path)
        table = MPC.BERT_MRPC_feature(
            topics=topic_list, cross_document=cross_document, cross_topic=cross_topic,
            positive_increase=0, shuffle=False, csv_path=BERT_csv_path, to_csv=to_csv)
        try:
            errors = table.loc[(table["#1 String"]==table["#2 String"]) & (table["Quality"]==0)]
            assert len(errors)==0
        except AssertionError as AE:
            print(errors.head())
