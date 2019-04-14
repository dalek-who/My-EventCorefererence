from itertools import combinations, product, chain
from tqdm import tqdm
import numpy as np
from pandas import DataFrame, Series, read_csv
import csv

from preprocessing.Structurize.EcbClass import *


class MentionPair(object):
    def __init__(self, mention1: EcbComponent, mention2: EcbComponent, ignore_order: bool=True):
        self.pair_id: str = ""
        self.component_pair: tuple = (None, None)

        m1_id = mention1.sentence.document.document_name + "-" + str(mention1.mid)
        m2_id = mention2.sentence.document.document_name + "-" + str(mention2.mid)
        pair_dict = {m1_id: mention1, m2_id: mention2}
        max_id, min_id = max(m1_id, m2_id), min(m1_id, m2_id)
        self.pair_id = max_id + min_id if ignore_order else m1_id + m2_id
        self.component_pair = (pair_dict[min_id], pair_dict[max_id]) if ignore_order else (mention1, mention2)

    def label(self, cross_document: bool=True) -> 0 or 1:
        if cross_document:
            return int(self.component_pair[0].instance_global == self.component_pair[1].instance_global)
        else:
            return int(self.component_pair[0].instance_within == self.component_pair[1].instance_within)

    def BERT_MRPC_feature(self, cross_document: bool=True):
        feature = Series()
        feature["Quality"] = self.label(cross_document=cross_document)
        feature["#1 ID"] = self.component_pair[0].sentence.sid()
        feature["#2 ID"] = self.component_pair[1].sentence.sid()
        feature["#1 String"] = self.component_pair[0].sentence.text()
        feature["#2 String"] = self.component_pair[1].sentence.text()
        return feature

    def sentences(self, text=True) -> tuple:
        if text:
            return (self.component_pair[0].sentence.text,
                    self.component_pair[1].sentence.text)
        else:
            return (self.component_pair[0].sentence,
                    self.component_pair[1].sentence)

    def document(self, text=True, drop_noise=True) -> tuple:
        pass

    def argument_vertex(self) -> tuple:
        pass


class MentionPairCreator(object):
    def __init__(self, ECBPLUS: EcbPlusTopView):
        self.ECBPLUS: EcbPlusTopView = ECBPLUS

    def generate_mention_pairs(self, topics: list or str="all", cross_document: bool=True, prefix_list: list=None,
                               ignore_order: bool=True, positive_increase: int=0, shuffle: bool=True):
        topics_list = list(self.ECBPLUS.document_view.topics_dict.keys()) if topics == "all" else topics[:]
        mention_pair_dict = dict()
        # 添加mention-pair
        for topic_id in tqdm(topics_list, desc="Generate mention-pair"):
            topic: EcbTopic = self.ECBPLUS.document_view.topics_dict[topic_id]
            mentions_in_doc_in_topic = []
            for document in topic.documents_dict.values():
                document: EcbDocument = document
                # 筛选文档中的mention
                if prefix_list:
                    mentions_in_doc = [comp for comp in document.components_dict.values() if
                                       comp.tag.startswith(tuple(prefix_list))]
                else:
                    mentions_in_doc = [comp for comp in document.components_dict.values()]
                mentions_in_doc_in_topic.append(mentions_in_doc)
            # 用来添加mention-pair
            def add_mention_pair(m1, m2, mention_pair_dict, ignore_order: bool=True):
                if m1 == m2:
                    return
                mention_pair = MentionPair(mention1=m1, mention2=m2, ignore_order=ignore_order)
                if mention_pair.pair_id in mention_pair_dict.keys():
                    pass
                else:
                    mention_pair_dict[mention_pair.pair_id] = mention_pair

            if cross_document:  # 文档内
                for m1, m2 in product(chain(*mentions_in_doc_in_topic), repeat=2):
                    add_mention_pair(m1, m2, mention_pair_dict, ignore_order=ignore_order)
            else:  # 文档间
                for mentions_in_doc in mentions_in_doc_in_topic:
                    for m1, m2 in product(mentions_in_doc, repeat=2):
                        add_mention_pair(m1, m2, mention_pair_dict, ignore_order=ignore_order)

        result = list(mention_pair_dict.values())

        # 扩增正例
        if positive_increase > 0:
            positive_samples = [pair for pair in result if pair.label(cross_document=cross_document)]
            result += positive_samples * positive_increase

        result = np.array(result)
        # shuffle
        if shuffle:
            np.random.shuffle(result)
        return result
