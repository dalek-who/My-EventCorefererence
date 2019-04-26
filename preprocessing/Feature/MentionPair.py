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

    def generate_sentence_pair(
            self,
            topics: list or str="all",
            selected: bool = True,
            cross_document: bool=True,
            cross_topic: bool=True,
            ignore_order: bool=True):

        topics_list = list(self.ECBPLUS.document_view.topics_dict.keys()) if topics == "all" else topics[:]
        sentences_all = []
        sentences_by_topic = dict()
        sentences_by_document = dict()
        # 搜出所有sentence
        for topic_id in topics_list:
            try:
                topic: EcbTopic = self.ECBPLUS.document_view.topics_dict[topic_id]
            except KeyError:
                print("no topic %s" % topic_id)
                continue
            sentences_by_topic[topic_id] = []
            for document in topic.documents_dict.values():
                document: EcbDocument = document
                if selected:
                    sentences_in_doc = [s for s in document.all_sentences_dict.values() if s.selected]
                else:
                    sentences_in_doc = [s for s in document.all_sentences_dict.values()]
                sentences_by_document[document.document_name] = sentences_in_doc
                sentences_by_topic[topic_id] += sentences_in_doc
                sentences_all += sentences_in_doc

        sentences_pairs = []
        if cross_topic:  # 跨document，跨topic
            pairs = combinations(sentences_all, r=2) if ignore_order else product(sentences_all, repeat=2)
            for s1, s2 in tqdm(pairs, desc="Generate sentence-pair [cross topic]:"):
                sentences_pairs.append((s1, s2))
        elif cross_document:  # 跨document，不跨topic
            for mentions_in_topic in tqdm(sentences_by_topic.values(), desc="Generate sentence-pair [cross document]:"):
                pairs = combinations(mentions_in_topic, r=2) if ignore_order else product(mentions_in_topic, repeat=2)
                for s1, s2 in pairs:
                    sentences_pairs.append((s1, s2))
        else:  # document内
            for mentions_in_doc in tqdm(sentences_by_document.values(), desc="Generate sentence-pair [within document]:"):
                pairs = combinations(mentions_in_doc, r=2) if ignore_order else product(mentions_in_doc, repeat=2)
                for s1, s2 in pairs:
                    sentences_pairs.append((s1, s2))
        return sentences_pairs

    def argument_feature(
            self,
            topics: str = "all",
            cross_document: bool = True,
            cross_topic: bool = True,
            positive_increase: int=0,
            shuffle: bool=True,
            csv_path: str="",
            to_csv: bool=True) -> DataFrame:
        sentences_pairs = self.generate_sentence_pair(
            topics=topics, selected=True, cross_document=cross_document, cross_topic=cross_topic, ignore_order=True)
        sentences_set = set(chain.from_iterable(sentences_pairs))
        sentences_instances = dict()
        # 生成每个句子的每类Argument集合
        for sentence in tqdm(sentences_set, desc="collecting arguments"):
            sentence: EcbSentence = sentence
            sentences_instances[sentence] = defaultdict(set)
            # 添加每个子类component的集合
            for component in sentence.components_dict.values():
                component: EcbComponent = component
                if component.tag.startswith("'UNKNOWN"):
                    continue
                instance = component.instance_global if cross_document else component.instance_within
                sentences_instances[sentence][component.tag].add(instance)
            # 添加time，loc，human_participate,non_human_prticipate,action,neg_action的集合
            actions = set()
            neg_actions = set()
            times = set()
            locations = set()
            human_participates = set()
            non_human_participates = set()
            for sub_tag, instances_set in sentences_instances[sentence].items():
                if sub_tag.startswith("ACTION"):
                    actions.update(instances_set)
                elif sub_tag.startswith("NEG_ACTION"):
                    neg_actions.update(instances_set)
                elif sub_tag.startswith("TIME"):
                    times.update(instances_set)
                elif sub_tag.startswith("LOC"):
                    locations.update(instances_set)
                elif sub_tag.startswith("HUMAN"):
                    human_participates.update(instances_set)
                elif sub_tag.startswith("NON_HUMAN"):
                    non_human_participates.update(instances_set)
                else:
                    raise KeyError(f"{sub_tag} is an illegal component tag")
            sentences_instances[sentence]["ACTION"].update(actions)
            sentences_instances[sentence]["NEG_ACTION"].update(neg_actions)
            sentences_instances[sentence]["TIME"].update(times)
            sentences_instances[sentence]["LOC"].update(locations)
            sentences_instances[sentence]["HUMAN"].update(human_participates)
            sentences_instances[sentence]["NON_HUMAN"].update(non_human_participates)

            sentences_instances[sentence]["ACTION_and_NEG_ACTION"].update(actions | neg_actions)
            sentences_instances[sentence]["HUMAN_and_NON_HUMAN"].update(human_participates | non_human_participates)
            sentences_instances[sentence]["Arguments"].update(times | locations | human_participates | non_human_participates)

        df_arg_feature = DataFrame(columns=["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"])
        for s1, s2 in tqdm(sentences_pairs, desc="generate records"):
            s1: EcbSentence = s1
            s2: EcbSentence = s2
            if s1.text() == s2.text():
                continue
            record = Series({"#1 ID":s1.sid(), "#2 ID":s2.sid(), "#1 String":s1.text(), "#2 String": s2.text()})
            record["Quality"] = int((sentences_instances[s1]["ACTION_and_NEG_ACTION"] & sentences_instances[s2]["ACTION_and_NEG_ACTION"]) != set())
            df_arg_feature = df_arg_feature.append(record, ignore_index=True)
        return df_arg_feature