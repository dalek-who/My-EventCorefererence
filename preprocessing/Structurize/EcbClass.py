""" 用于结构化ECB+数据集的类 """
import os
from collections import defaultdict
from pandas import read_csv
import xml.etree.ElementTree as et

from preprocessing.Structurize.utils import *

ECB_DRI = "../../datasets/ECB+_LREC2014/"
CSV_DIR = ECB_DRI + "ECBplus_coreference_sentences.csv"
DATA_DIR = ECB_DRI + "ECB+/"
CSV_ECBplus_coreference_sentences = read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), CSV_DIR)))

class EcbPlusTopView(object):
    """ ECB+数据集最顶层的视角"""
    def __init__(self):
        self.document_view = None
        self.coreference_view = None


class EcbDocumentView(object):
    def __init__(self):
        # data
        self.topics_dict: dict = {}
        self.path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_DIR))
        # generate data
        topic_ids = sorted(int(topic_id) for topic_id in os.listdir(self.path))
        self.topics_dict = {topic_id: EcbTopic(topic_id) for topic_id in topic_ids}


class EcbCoreferenceView(object):
    def __init__(self):
        pass


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
        self.tokens_list: list = []
        self.components_list: list = []
        self.instances_list: list = []
        self.relations_list: list = []

        # generate properties
        # read xml files
        tree = et.parse(source=self.path)
        root = tree.getroot()
        self.doc_id = root.get("doc_id")
        # get tokens
        tokens = root.findall("token")
        tokens = [EcbToken(word=token.text,
                           sentence=int(token.get("sentence")),
                           tid=int(token.get("t_id")),
                           number=int(token.get("number")),
                           mid=None) for token in tokens]
        # get components
        markables = root.find("Markables").getchildren()
        components = [EcbComponent(tag=mark.tag,
                                   mid=int(mark.get("m_id")),
                                   anchor_tid_list=[int(anchor.get("t_id")) for anchor in mark.findall("token_anchor")],
                                   note=mark.get("note"))
                      for mark in markables if mark.get("instance_id") is None]
        instances = [EcbInstance(tag=mark.tag,
                                 mid=mark.get("m_id"),
                                 related_to=mark.get("RELATED_TO"),
                                 description=mark.get("TAG_DESCRIPTOR"),
                                 iid=mark.get("instance_id"))
                     for mark in markables if mark.get("instance_id") is not None]
        relations = root.find("Relations").getchildren()
        relations = [EcbCorefRelation(tag=relation.tag,
                                      rid=int(relation.get("r_id")),
                                      note=relation.get("note"),
                                      sources_mid_list=[int(source.get("m_id")) for source in relation.findall("source")],
                                      target_mid=int(relation.find("target").get("m_id")))
                     for relation in relations]
        self.tokens_list = tokens
        self.components_list = components
        self.instances_list = instances
        self.relations_list = relations


class EcbToken(object):
    def __init__(self, word: str, sentence: int, tid: int, number:int, mid: int=None):
        """
        example:
        <token t_id="49" sentence="1" number="38">Treatment</token>
        """
        self.word: str = word
        self.sentence: int = sentence
        self.tid: int = tid
        self.number: int = number
        self.mid: int = mid


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


class EcbInstance(object):
    def __init__(self,tag: str, mid:int, related_to: str, description: str, iid: str):
        # example:
        # <ACTION_OCCURRENCE m_id="50" RELATED_TO="" TAG_DESCRIPTOR="t1b_checking_in_promises" instance_id="ACT16236402809085484" />
        self.tag: str = tag
        self.mid: int = mid
        self.related_to: str = related_to
        self.description: str = description
        self.iid: str = iid


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


class EcbSentence(object):
    def __init__(self):
        pass


class EcbCluster(object):
    def __init__(self):
        pass


class EcbMention(object):
    def __init__(self):
        pass

