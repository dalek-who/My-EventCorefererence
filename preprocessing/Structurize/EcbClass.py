""" 用于结构化ECB+数据集的类 """
import sys
import os
import copy
from collections import defaultdict
from pandas import read_csv, DataFrame
import xml.etree.ElementTree as et
from typing import List, Dict
from itertools import chain

from preprocessing.Structurize.utils import *
from configs import CONFIG

ECB_DRI = "../../datasets/ECB+_LREC2014/"
CSV_DIR = ECB_DRI + "ECBplus_coreference_sentences.csv"
DATA_DIR = ECB_DRI + "ECB+/"
CSV_ECBplus_coreference_sentences = read_csv(CONFIG.CSV_DIR)


class EcbPlusTopView(object):
    """ ECB+数据集最顶层的视角"""
    def __init__(self):
        self.document_view = EcbDocumentView()
        self.coreference_view = EcbCoreferenceView(DodumentView=self.document_view)

    def BERT_MRPC_Data(self, topics: list or str="all", cross_document: bool=True):
        dataset = []
        topic_list = self.coreference_view.instance_dict_by_topic.keys() if topics=="all" else topics
        # todo


class EcbDocumentView(object):
    def __init__(self):
        # data
        self.topics_dict: dict = {}
        self.path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_DIR))
        # generate data
        topic_ids = sorted(int(topic_id) for topic_id in os.listdir(self.path))
        self.topics_dict = {topic_id: EcbTopic(topic_id) for topic_id in topic_ids}


class EcbCoreferenceView(object):
    def __init__(self, DodumentView: EcbDocumentView):
        self.instance_dict_by_topic: dict = {}
        # 产生全局的Coreference簇
        for tid, topic in DodumentView.topics_dict.items():
            topic: EcbTopic = topic
            self.instance_dict_by_topic[tid] = {}
            for document in topic.documents_dict.values():
                document: EcbDocument = document
                for iid, instance in document.instances_dict.items():
                    if iid not in self.instance_dict_by_topic[tid].keys():
                        self.instance_dict_by_topic[tid][iid] = copy.copy(instance)
                        self.instance_dict_by_topic[tid][iid].mentions_list = instance.mentions_list[:]
                    else:
                        self.instance_dict_by_topic[tid][iid].mentions_list += instance.mentions_list[:]
                    self.instance_dict_by_topic[tid][iid].singleton_global = len(self.instance_dict_by_topic[tid][iid].mentions_list) < 2
                    if len(self.instance_dict_by_topic[tid][iid].mentions_list) == 0:
                        print("EcbInstance iid=%s, mid=%s has no components" % (self.instance_dict_by_topic[tid][iid].iid, self.instance_dict_by_topic[tid][iid].mid))
        # 建立component到全局instance的反向索引
        for topic_id, topic in self.instance_dict_by_topic.items():
            for iid, instance in topic.items():
                instance: EcbInstance = instance
                for component in instance.mentions_list:
                    component: EcbComponent = component
                    component.instance_global = instance


class EcbTopic(object):
    def __init__(self, topic_id: int):
        # properties
        self.topic_id: int = topic_id
        self.path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_DIR, str(self.topic_id)))
        self.documents_dict: dict = {}
        # generate properties
        file_list = os.listdir(self.path)
        file_list = sorted([EcbFilename2Tuple(file_name) for file_name in file_list])
        self.documents_dict = {Tuple2EcbFilename(file_tuple): EcbDocument(topic_id=self.topic_id,
                                                                          document_name=Tuple2EcbFilename(file_tuple),
                                                                          category=file_tuple[-1])
                               for file_tuple in file_list}


class EcbDocument(object):
    def __init__(self, topic_id: int, document_name: str, category: str):
        # properties
        self.topic_id: int = topic_id
        self.document_name: str = document_name
        self.doc_id: str = None
        self.category: str = category
        self.path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_DIR, str(self.topic_id), self.document_name))
        self.all_sentences_dict: dict = {}
        self.tokens_dict: dict = self.parse_tokens_dict()
        self.components_dict: dict = self.parse_components_dict()
        self.instances_dict: dict = self.parse_instances_dict()

        # generate properties
        # read xml files
        tree = et.parse(source=self.path)
        root = tree.getroot()
        self.doc_id = root.get("doc_id")

        # 创建sentence，把token添加到对应的sentence中,建立sentence与token之间的互相索引
        tokens_dict = self.tokens_dict
        for tid, token in tokens_dict.items():
            token: EcbToken = token
            if self.all_sentences_dict.get(token.sentence_id) is None:
                self.all_sentences_dict[token.sentence_id] = EcbSentence(
                    document_name=self.document_name, sentence_id=token.sentence_id, document=self)
            sentence: EcbSentence = self.all_sentences_dict[token.sentence_id]
            sentence.tokens_list.append(token)
            token.sentence = sentence  # token到sentence的反向索引

        # 把component添加到对应句子中, 建立sentence与component之间的互相索引
        components_dict = self.components_dict
        mid: int
        component: EcbComponent
        for mid, component in components_dict.items():
            component_tokens: List[EcbToken] = [self.tokens_dict[tid] for tid in component.anchor_tid_list]
            # 把component对应的text添加给每个component
            text = " ".join([token.word for token in component_tokens])
            component.text = text
            # 把component添加到对应句子中
            # 有一些类型为UNKNOWN的component，其token list是空的
            if len(component_tokens) > 0:
                sentence: EcbSentence = self.all_sentences_dict[component_tokens[0].sentence_id]
                sentence.components_dict[mid] = component
                component.sentence = sentence  # component到sentence的反向索引

        # 建立token与component之间的互相索引
        for mid, component in self.components_dict.items():
            component: EcbComponent = component
            component.tokens_list: list = [self.tokens_dict[tid] for tid in component.anchor_tid_list]
            for token in component.tokens_list:
                token: EcbToken = token
                token.component = component  # token到component的反向索引

        # 选取被选择到coreference数据集中的句子
        Topic = self.topic_id
        File = self.document_name.rstrip(".xml").split("_")[-1]
        try:
            selected_sentence_id = CSV_ECBplus_coreference_sentences.groupby(["Topic", "File"]).get_group((Topic, File))["Sentence Number"]
            for sid in selected_sentence_id:
                sentence: EcbSentence = self.all_sentences_dict[sid]
                sentence.selected = True
        except KeyError as KE:
            print("No sentences are selected in this file:", KE)

        # 创建内部聚类簇（有些聚类簇在文档内只有一个component，但跨文档不止一个component），建立component与内部instance互相的索引
        instances_dict_with_mid_key: dict = {instance.mid: instance for instance in self.instances_dict.values()}
        relations_dict = self.parse_relations_dict()
        for rid, relation in relations_dict.items():
            relation: EcbCorefRelation = relation
            source_components = [self.components_dict[src_mid] for src_mid in relation.sources_mid_list]
            instance: EcbInstance = instances_dict_with_mid_key[relation.target_mid]
            instance.mentions_list += source_components
            for component in instance.mentions_list:
                component.instance_within = instance  # component到内部instance的反向索引

        # 创建全局Singleton，建立component与instance互相的索引
        non_singleton_mid_set = set(chain(*[relation.sources_mid_list for relation in relations_dict.values()]))
        all_mid_set = set(self.components_dict.keys())
        singleton_mid_set = all_mid_set - non_singleton_mid_set
        for i, mid in enumerate(singleton_mid_set):
            singleton_iid = "Singleton_%s_%s" % (self.document_name, i)
            component: EcbComponent = self.components_dict[mid]
            instance: EcbInstance = EcbInstance(tag=component.tag,
                                                mid=None,
                                                related_to="",
                                                description="",
                                                iid=singleton_iid,
                                                singleton_global=True)
            instance.mentions_list.append(component)
            self.instances_dict[singleton_iid] = instance
            for component in instance.mentions_list:
                component.instance_within = instance  # component到内部instance的反向索引

        # 识别哪些是文档内singleton
        for instance in self.instances_dict.values():
            instance: EcbInstance = instance
            instance.singleton_within = len(instance.mentions_list) < 2
            if len(instance.mentions_list)==0:
                print("EcbInstance iid=%s, mid=%s in %s has no components" % (instance.iid, instance.mid, self.document_name))

    def parse_tokens_dict(self) -> dict:
        tree = et.parse(source=self.path)
        root = tree.getroot()
        self.doc_id = root.get("doc_id")
        # get tokens
        tokens = root.findall("token")
        tokens = {int(token.get("t_id")): EcbToken(word=token.text,
                                                   sentence_id=int(token.get("sentence")),
                                                   tid=int(token.get("t_id")),
                                                   number=int(token.get("number")),
                                                   mid=None)
                  for token in tokens}
        return tokens

    def parse_components_dict(self) -> dict:
        tree = et.parse(source=self.path)
        root = tree.getroot()
        markables = root.find("Markables").getchildren()
        components = {int(mark.get("m_id")): EcbComponent(tag=mark.tag,
                                                          mid=int(mark.get("m_id")),
                                                          anchor_tid_list=[int(anchor.get("t_id")) for anchor in mark.findall("token_anchor")],
                                                          note=mark.get("note"))
                      for mark in markables if mark.get("RELATED_TO") is None}
        return components

    def parse_instances_dict(self) -> dict:
        tree = et.parse(source=self.path)
        root = tree.getroot()
        markables = root.find("Markables").getchildren()
        def get_iid(mark):
            if mark.get("instance_id") is not None:
                return mark.get("instance_id")
            else:
                return "%s_%s_%s" % (mark.tag, self.document_name, mark.get("m_id"))

        instances = {get_iid(mark): EcbInstance(tag=mark.tag,
                                                mid=int(mark.get("m_id")),
                                                related_to=mark.get("RELATED_TO"),
                                                description=mark.get("TAG_DESCRIPTOR"),
                                                iid=get_iid(mark),
                                                singleton_global=False)
                     for mark in markables if mark.get("RELATED_TO") is not None}
        return instances

    def parse_relations_dict(self):
        tree = et.parse(source=self.path)
        root = tree.getroot()
        relations = root.find("Relations").getchildren()
        relations = {int(relation.get("r_id")): EcbCorefRelation(tag=relation.tag,
                                                                 rid=int(relation.get("r_id")),
                                                                 note=relation.get("note"),
                                                                 sources_mid_list=[int(source.get("m_id")) for source in relation.findall("source")],
                                                                 target_mid=int(relation.find("target").get("m_id")))
                     for relation in relations}
        return relations

    def singletons_within(self):
        # 内部的singleton: 全局singleton+在此document中只有一个component的Instance
        pass

    def clusters_within(self):
        # 内部的聚类簇：在此document中有至少两个component的Instance
        pass


class EcbSentence(object):
    def __init__(self, document_name: str, sentence_id: int, document: EcbDocument):
        self.document_name: str = document_name
        self.sentence_id: int = sentence_id
        self.selected: bool = False
        self.tokens_list: List[EcbToken] = []
        self.components_dict: Dict[int, EcbComponent] = {}
        self.document: EcbDocument = document

    def sid(self):
        return self.document.document_name + "-" + str(self.sentence_id)

    def text(self):
        return " ".join([token.word for token in self.tokens_list])


class EcbToken(object):
    def __init__(self, word: str, sentence_id: int, tid: int, number: int, mid: int=None):
        """
        example:
        <token t_id="49" sentence="1" number="38">Treatment</token>
        """
        self.word: str = word
        self.sentence_id: int = sentence_id
        self.tid: int = tid
        self.number: int = number
        self.mid: int = mid
        self.component: EcbComponent = None
        self.sentence: EcbSentence = None


class EcbComponent(object):
    def __init__(self, tag: str, mid: int, anchor_tid_list: list, note: str=None):
        # exmple:
        # <HUMAN_PART_ORG m_id="38" note="byCROMER" >  # note might be None
        #     <token_anchor t_id="14"/>
        #     <token_anchor t_id="15"/>
        # </HUMAN_PART_ORG>
        self.tag: str = tag
        self.mid: int = mid
        self.note: str = note
        self.anchor_tid_list: list = anchor_tid_list
        self.text: str = None
        self.sentence: EcbSentence = None
        self.tokens_list: list = []
        self.instance_within: EcbInstance = None
        self.instance_global: EcbInstance = None

    def global_mid(self) -> str:
        return self.sentence.document.document_name + "-" + str(self.mid)


class EcbInstance(object):
    def __init__(self, tag: str, mid: int, related_to: str, description: str, iid: str, singleton_global: bool):
        # example:
        # <ACTION_OCCURRENCE m_id="50" RELATED_TO="" TAG_DESCRIPTOR="t1b_checking_in_promises" instance_id="ACT16236402809085484" />
        self.tag: str = tag
        self.mid: int = mid
        self.related_to: str = related_to
        self.description: str = description
        self.iid: str = iid
        self.singleton_global: bool = singleton_global  # 是否是全局singleton
        self.singleton_within: bool = None  # 是否是文档内singleton
        self.mentions_list: list = []


class EcbCorefRelation(object):
    def __init__(self,tag:str, rid: int, note: str, sources_mid_list: list, target_mid: int):
        # example:
        #  < CROSS_DOC_COREF  r_id = "37542"  note = "HUM16236907954762763" >
        #       < source  m_id = "38" / >
        #       < source  m_id = "42" / >
        #       < target  m_id = "47" / >
        #  < / CROSS_DOC_COREF >
        self.rid: int = rid
        self.note: str = note
        self.sources_mid_list: list = sources_mid_list
        self.target_mid: int = target_mid



class EcbCluster(object):
    def __init__(self):
        pass


class EcbMention(object):
    def __init__(self):
        pass

