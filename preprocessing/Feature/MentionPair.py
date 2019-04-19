import sys
sys.path.append("../..")

from itertools import combinations, product, chain
from tqdm import tqdm
import numpy as np
from pandas import DataFrame, Series, read_csv
import csv

from preprocessing.Structurize.EcbClass import *


class MentionPair(object):
    def __init__(self, mention1: EcbComponent, mention2: EcbComponent, ignore_order: bool=True):
        self.component_pair: tuple = (None, None)
        self.fixed_label = None

        m1_id = mention1.global_mid()
        m2_id = mention2.global_mid()
        pair_dict = {m1_id: mention1, m2_id: mention2}
        max_id, min_id = max(m1_id, m2_id), min(m1_id, m2_id)
        self.component_pair = (pair_dict[min_id], pair_dict[max_id]) if ignore_order else (mention1, mention2)

    def label(self, cross_document: bool=True) -> 0 or 1:
        if cross_document:
            return int(self.component_pair[0].instance_global == self.component_pair[1].instance_global)
        else:
            return int(self.component_pair[0].instance_within == self.component_pair[1].instance_within)

    def pair_id(self,by_what: str="mention"):
        if by_what=="mention":
            return self.component_pair[0].global_mid() + "+" +\
                   self.component_pair[1].global_mid()
        elif by_what=="sentence":
            return self.component_pair[0].sentence.sid() + "+" + \
                   self.component_pair[1].sentence.sid()
        elif by_what=="document":
            return self.component_pair[0].sentence.document.document_name + "+" + \
                   self.component_pair[1].sentence.document.document_name
        else:
            raise ValueError("by_what=%s is invalid" % by_what)

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

    def generate_mention_pairs(
            self,
            topics: list or str="all",
            cross_document: bool=True,
            cross_topic: bool=True,
            prefix_list: list=None,
            ignore_order: bool=True,
            positive_increase: int=0,
            shuffle: bool=True,
            by_what: str="mention"):
        topics_list = list(self.ECBPLUS.document_view.topics_dict.keys()) if topics == "all" else topics[:]
        mention_pair_dict = dict()

        # 用来添加mention-pair 的函数
        def add_mention_pair(m1, m2, mention_pair_dict, ignore_order: bool = True, by_what: str="mention"):
            if by_what=="sentence" and m1.sentence.text() == m2.sentence.text():
                return
            mention_pair = MentionPair(mention1=m1, mention2=m2, ignore_order=ignore_order)
            if mention_pair.pair_id(by_what=by_what) in mention_pair_dict.keys():
                exist_pair = mention_pair_dict[mention_pair.pair_id(by_what=by_what)]
                if exist_pair.fixed_label in (None, 0) and mention_pair.label(cross_document=cross_document) ==1:
                    exist_pair.fixed_label = 1
            else:
                mention_pair_dict[mention_pair.pair_id(by_what=by_what)] = mention_pair
                mention_pair.fixed_label = mention_pair.label(cross_document=cross_document)

        mentions_all = []  # 所有mention
        mentions_by_topic = {}  # 每个topic中的mention
        mentions_by_document = {}  # 每个文档中的mention
        # 搜出所有mention
        for topic_id in topics_list:
            try:
                topic: EcbTopic = self.ECBPLUS.document_view.topics_dict[topic_id]
            except KeyError:
                print("no topic %s" % topic_id)
                continue
            mentions_by_topic[topic_id] = []
            for document in topic.documents_dict.values():
                document: EcbDocument = document
                mentions_by_document[document.document_name] = []
                # 筛选文档中的mention
                if prefix_list:
                    mentions_in_doc = [comp for comp in document.components_dict.values() if
                                       comp.tag.startswith(tuple(prefix_list))]
                else:
                    mentions_in_doc = [comp for comp in document.components_dict.values()]
                mentions_by_document[document.document_name] = mentions_in_doc
                mentions_by_topic[topic_id] += mentions_in_doc
                mentions_all += mentions_in_doc

        # 生成mention pair
        if cross_topic:  # 跨document，跨topic
            for m1, m2 in tqdm(product(mentions_all, repeat=2), desc="Generate mention-pair [cross topic]:"):
                add_mention_pair(m1, m2, mention_pair_dict, ignore_order=ignore_order, by_what=by_what)
        elif cross_document:  # 跨document，不跨topic
            for mentions_in_topic in tqdm(mentions_by_topic.values(), desc="Generate mention-pair [cross document]:"):
                for m1, m2 in product(mentions_in_topic, repeat=2):
                    add_mention_pair(m1, m2, mention_pair_dict, ignore_order=ignore_order, by_what=by_what)
        else:  # document内
            for mentions_in_doc in tqdm(mentions_by_document.values(), desc="Generate mention-pair [within document]:"):
                for m1, m2 in product(mentions_in_doc, repeat=2):
                    add_mention_pair(m1, m2, mention_pair_dict, ignore_order=ignore_order, by_what=by_what)
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

    def BERT_MRPC_feature(
            self,
            topics: str = "all",
            cross_document: bool = True,
            cross_topic: bool = True,
            positive_increase: int=0,
            shuffle: bool=True,
            csv_path: str="",
            to_csv: bool=True) -> DataFrame:
        mention_pair_array: np.array = self.generate_mention_pairs(
            topics=topics, cross_document=cross_document, cross_topic=cross_topic, prefix_list=["ACTION", "NEG_ACTION"],
            ignore_order=True, positive_increase=positive_increase, shuffle=shuffle, by_what="sentence")
        # table = DataFrame(columns=["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"])
        # for pair in mention_pair_array:
        #     pair: MentionPair = pair
        #     if table.empty \
        #             or pair.pair_id(by_what="sentence") not in table.index \
        #             or table.loc[pair.pair_id(by_what="sentence"), "Quality"] == 0 and pair.label(cross_document=cross_document)==1:
        #         new_data = Series({
        #             "Quality": pair.label(cross_document=cross_document),
        #             "#1 ID": pair.component_pair[0].sentence.sid(),
        #             "#2 ID": pair.component_pair[1].sentence.sid(),
        #             "#1 String": pair.component_pair[0].sentence.text(),
        #             "#2 String": pair.component_pair[1].sentence.text(),
        #         })
        #         table.loc[pair.pair_id(by_what="sentence"), :] = new_data
        table = DataFrame()
        table["Quality"] = Series([pair.fixed_label for pair in mention_pair_array])
        table["#1 ID"] = Series([pair.component_pair[0].sentence.sid() for pair in mention_pair_array])
        table["#2 ID"] = Series([pair.component_pair[1].sentence.sid() for pair in mention_pair_array])
        table["#1 String"] = Series([pair.component_pair[0].sentence.text() for pair in mention_pair_array])
        table["#2 String"] = Series([pair.component_pair[1].sentence.text() for pair in mention_pair_array])
        if to_csv:
            table.to_csv(csv_path, sep="\t", index=False)
        return table
