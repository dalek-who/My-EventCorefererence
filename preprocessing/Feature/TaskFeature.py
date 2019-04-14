from pandas import DataFrame, Series
import numpy as np
import csv

from preprocessing.Structurize.EcbClass import *
from preprocessing.Feature.MentionPair import MentionPair, MentionPairCreator


class BERT_MRPC_FeatureCreator(object):
    def __init__(self):
        self.mention_pair_creator = MentionPairCreator(EcbPlusTopView())

    def feature_csv(self,
                    csv_path: str,
                    topics: str = "all",
                    cross_document: bool = True,
                    positive_increase: int=0,
                    shuffle: bool=True) -> DataFrame:
        mention_pair_array: np.array = self.mention_pair_creator.generate_mention_pairs(
            topics=topics, cross_document=cross_document, prefix_list=["ACTION", "NEG_ACTION"],
            ignore_order=True, positive_increase=positive_increase, shuffle=shuffle)
        table = DataFrame()
        table["Quality"] = Series([pair.label(cross_document=cross_document) for pair in mention_pair_array])
        table["#1 ID"] = Series([pair.component_pair[0].sentence.sid() for pair in mention_pair_array])
        table["#2 ID"] = Series([pair.component_pair[1].sentence.sid() for pair in mention_pair_array])
        table["#1 String"] = Series([pair.component_pair[0].sentence.text() for pair in mention_pair_array])
        table["#2 String"] = Series([pair.component_pair[1].sentence.text() for pair in mention_pair_array])
        # table.to_csv(csv_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)
        table.to_csv(csv_path, sep="\t", index=False)
        return table
