import sys
sys.path.append("../..")

from itertools import combinations, product, chain
from tqdm import tqdm
import numpy as np
from pandas import DataFrame, Series, read_csv
import csv
import networkx as nx
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from torch_geometric.data import Data as GnnData
from math import log
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

from preprocessing.Structurize.EcbClass import *


class MentionPair(object):
    def __init__(self, mention1: EcbComponent, mention2: EcbComponent, ignore_order: bool=True):
        self.mention_pair: tuple = (None, None)

        m1_id = mention1.global_mid()
        m2_id = mention2.global_mid()
        pair_dict = {m1_id: mention1, m2_id: mention2}
        max_id, min_id = max(m1_id, m2_id), min(m1_id, m2_id)
        self.mention_pair = (pair_dict[min_id], pair_dict[max_id]) if ignore_order else (mention1, mention2)

    def label(self, cross_document: bool=True) -> 0 or 1:
        if cross_document:
            return int(self.mention_pair[0].instance_global == self.mention_pair[1].instance_global)
        else:
            return int(self.mention_pair[0].instance_within == self.mention_pair[1].instance_within)

    def pair_id(self):
        return self.mention_pair[0].global_mid() + "+" + \
               self.mention_pair[1].global_mid()

    def argument_graph(self):
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
            ignore_order: bool=True):
        topics_list = list(self.ECBPLUS.document_view.topics_dict.keys()) if topics == "all" else topics[:]
        mention_pair_dict = dict()

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
        mention_pair_dict = dict()
        if cross_topic:  # 跨document，跨topic
            pair_iter = combinations(mentions_all, r=2) if ignore_order else product(mentions_all, repeat=2)
            for m1, m2 in tqdm(pair_iter, desc="Generate mention-pair [cross topic]:"):
                if not (m1.sentence.selected and m2.sentence.selected):
                    continue
                pair = MentionPair(m1, m2, ignore_order=ignore_order)
                if ignore_order:
                    assert mention_pair_dict.get(pair.pair_id()) is None
                mention_pair_dict[pair.pair_id()] = pair
        elif cross_document:  # 跨document，不跨topic
            for mentions_in_topic in tqdm(mentions_by_topic.values(), desc="Generate mention-pair [cross document]:"):
                pair_iter = combinations(mentions_in_topic, r=2) if ignore_order else product(mentions_in_topic, repeat=2)
                for m1, m2 in pair_iter:
                    if not (m1.sentence.selected and m2.sentence.selected):
                        continue
                    pair = MentionPair(m1, m2, ignore_order=ignore_order)
                    if ignore_order:
                        assert mention_pair_dict.get(pair.pair_id()) is None
                    mention_pair_dict[pair.pair_id()] = pair
        else:  # document内
            for mentions_in_doc in tqdm(mentions_by_document.values(), desc="Generate mention-pair [within document]:"):
                pair_iter = combinations(mentions_in_doc, r=2) if ignore_order else product(mentions_in_doc, repeat=2)
                for m1, m2 in pair_iter:
                    if not (m1.sentence.selected and m2.sentence.selected):
                        continue
                    pair = MentionPair(m1, m2, ignore_order=ignore_order)
                    if ignore_order:
                        assert mention_pair_dict.get(pair.pair_id()) is None
                    mention_pair_dict[pair.pair_id()] = pair

        result = list(mention_pair_dict.values())
        return result


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(
            self,
            id_a,
            id_b,
            label,
            input_ids,
            input_mask,
            segment_ids,
            trigger_mask_a,
            trigger_mask_b,
            tf_idf_a,
            tf_idf_b,
            gnn_data
    ):
        self.id_a = id_a
        self.id_b = id_b
        self.label = label
        # bert 句子特征的输入
        self.BERT_input_ids = input_ids
        self.BERT_input_mask = input_mask
        self.BERT_segment_ids = segment_ids
        # bert 提取trigger上下文的范围
        self.BERT_trigger_mask_a = trigger_mask_a
        self.BERT_trigger_mask_b = trigger_mask_b

        # TF-IDF
        self.tfidf = np.concatenate((tf_idf_a, tf_idf_b), axis=0)

        # 图网络特征
        self.gnn_data = gnn_data


class InputFeaturesCreator(object):
    def __init__(self, ECBPlus: EcbPlusTopView):
        self.ecb = ECBPlus

    # def load(self,
    #          cross_document: bool=True,
    #          cross_topic: bool=False):
    #     cross_document_str = "_cross_doc" if cross_document else "_within_doc"
    #     cross_topic_str = "_cross_topic" if cross_topic else ""
    #     file_name = "./feature_%s%s.pkl" % (cross_document_str, cross_topic_str)
    #     path = os.path.join(os.path.dirname(__file__), file_name)
    #     with open(path, "rb") as f:
    #         features = pickle.load(f)
    #     return features

    def create_from_dataset(
            self,
            topics: list or str="all",
            cross_document: bool=True,
            cross_topic: bool=False,
            ignore_order: bool=True,
            max_seq_length: int=123,
            trigger_half_window: int=1
    ):

        mention_pair_list = MentionPairCreator(self.ecb).generate_mention_pairs(
            topics=topics, cross_document=cross_document, cross_topic=cross_topic, prefix_list=["ACTION", "NEG_ACTION"], ignore_order=ignore_order)
        features = []
        for pair in tqdm(mention_pair_list, desc="Extracting features:"):
            pair: MentionPair = pair
            input_ids, input_mask, segment_ids, trigger_mask_a, trigger_mask_b = \
                convert_mention_pair_to_BERT_features(mention_pair=pair, max_seq_length=max_seq_length, trigger_half_window=trigger_half_window)
            gnn_data = convert_mention_pair_to_GNN_features(mention_pair=pair, cross_document=cross_document)
            feature = InputFeatures(
                id_a=pair.mention_pair[0].global_mid(),
                id_b=pair.mention_pair[1].global_mid(),
                label=int(pair.label(cross_document=cross_document)),
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                trigger_mask_a=trigger_mask_a,
                trigger_mask_b=trigger_mask_b,
                tf_idf_a=pair.mention_pair[0].sentence.document.tfidf,
                tf_idf_b=pair.mention_pair[1].sentence.document.tfidf,
                gnn_data = gnn_data
            )
            features.append(feature)

        cross_document_str = "_cross_doc" if cross_document else "_within_doc"
        cross_topic_str = "_cross_topic" if cross_topic else ""
        file_name = "./feature_%s%s.pkl" % (cross_document_str, cross_topic_str)
        path = os.path.join(os.path.dirname(__file__), file_name)
        with open(path, "wb") as f:
            pickle.dump(features, f)
        return features


def find_word_piece_index_for_tokens(tokens: list, word_pieces: list):
    i, j = 0, 0
    token_wordpiece_list = []
    start_a_new_token = True
    while i < len(tokens) and j < len(word_pieces):
        current_token = tokens[i].lower().rstrip("\t").replace("�","").replace("ô","o") # 数据集中这些符号在tokenlize时会变得和原来不一样，直接去掉
        if start_a_new_token:
            new_token = ""
            new_pair = [tokens[i], i, [], []]
            new_record = {"token": tokens[i], "token_id": i, "word_pieces": [], "word_pieces_ids":[]}
        new_piece = word_pieces[j] if not word_pieces[j].startswith("##") else word_pieces[j][2:]
        new_token += new_piece
        if new_token == current_token:
            new_record["word_pieces"].append(word_pieces[j])
            new_record["word_pieces_ids"].append(j)
            token_wordpiece_list.append(new_record)
            i += 1
            start_a_new_token = True
        else:
            new_record["word_pieces"].append(word_pieces[j])
            new_record["word_pieces_ids"].append(j)
            start_a_new_token = False
        j += 1
    assert i == len(tokens) and j == len(word_pieces) and len(token_wordpiece_list)==len(tokens)
    return token_wordpiece_list


def convert_mention_pair_to_BERT_features(mention_pair: MentionPair, max_seq_length: int, trigger_half_window: int):
    if len(mention_pair.mention_pair[0].tokens_list) >1 or len(mention_pair.mention_pair[1].tokens_list) >1:
        pass
    """Loads a data file into a list of `InputBatch`s."""

    word_pieces = {"a":[], "b":[]}  # 两个mention句子tokenlize后的word_piece
    trigger_range_lists = {"a_min": -1, "a_max": -1, "b_min": -1, "b_max": -1}  # 转换为wordpiece后，两个trigger附近上下文的最左、最右wordpiece id

    for which, mention in [("a", mention_pair.mention_pair[0]), ("b", mention_pair.mention_pair[1])]:
        tokens = [tok.word for tok in mention.sentence.tokens_list]
        text = mention.sentence.text().rstrip("\t").replace("�", "").replace("ô", "o")
        # word_pieces[which] = tokenizer.tokenize(text)
        word_pieces[which] = mention.sentence.word_pieces_list[:]
        token_wordpiece_list = find_word_piece_index_for_tokens(tokens, word_pieces[which])
        first_trigger_token_id = max(mention.tokens_list[0].number - trigger_half_window, 0)
        last_trigger_token_id = min(mention.tokens_list[-1].number + trigger_half_window, len(tokens)-1)
        first_trigger_word_piece_id = token_wordpiece_list[first_trigger_token_id]["word_pieces_ids"][0]
        last_trigger_word_piece_id  = token_wordpiece_list[last_trigger_token_id]["word_pieces_ids"][-1]
        _truncate_seq_pair(word_pieces["a"], word_pieces["b"], max_seq_length - 3)
        last_trigger_word_piece_id = min(last_trigger_word_piece_id, len(word_pieces[which])-1)
        trigger_range_lists[which+"_min"] = first_trigger_word_piece_id
        trigger_range_lists[which+"_max"] = last_trigger_word_piece_id

    word_pieces_a, word_pieces_b = word_pieces["a"], word_pieces["b"]
    # trigger上下文范围
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + word_pieces_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    trigger_range_left_a = trigger_range_lists["a_min"] +1
    trigger_range_right_a = trigger_range_lists["a_max"] +1 + 1
    trigger_range_left_b = trigger_range_lists["b_min"] + len(tokens)
    trigger_range_right_b = trigger_range_lists["b_max"] + len(tokens) + 1

    tokens += word_pieces_b + ["[SEP]"]
    segment_ids += [1] * (len(word_pieces_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    trigger_mask_a = [1 if trigger_range_left_a <= i < trigger_range_right_a else 0 for i in range(max_seq_length) ]
    trigger_mask_b = [1 if trigger_range_left_b <= i < trigger_range_right_b else 0 for i in range(max_seq_length)]
    return input_ids, input_mask, segment_ids, trigger_mask_a, trigger_mask_b


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def trigger_argument_distance(trigger: EcbComponent, argument: EcbComponent):
    # argument 到 trigger 的最短距离
    if trigger.tokens_list[0].tid <= argument.tokens_list[0].tid <= trigger.tokens_list[-1].tid \
            or argument.tokens_list[0].tid <= trigger.tokens_list[0].tid <= argument.tokens_list[-1].tid:
        return 0
    else:
        return min(
            abs(trigger.tokens_list[0].tid - argument.tokens_list[0].tid),
            abs(trigger.tokens_list[0].tid - argument.tokens_list[-1].tid),
            abs(trigger.tokens_list[-1].tid - argument.tokens_list[0].tid),
            abs(trigger.tokens_list[-1].tid - argument.tokens_list[-1].tid),
        )

def normalize_edge_weight(distance: int):
    return log(1 + 1/(0.1 + distance))  # log normalize, 距离越远越不重要，0.1防止分母为0


def gnn_graph(m1: EcbComponent, m2: EcbComponent, cross_document: bool):
    tags = {
        'ACTION_ASPECTUAL', 'ACTION_CAUSATIVE', 'ACTION_GENERIC', 'ACTION_OCCURRENCE', 'ACTION_PERCEPTION',
    'ACTION_REPORTING', 'ACTION_STATE',
        'NEG_ACTION_ASPECTUAL', 'NEG_ACTION_CAUSATIVE', 'NEG_ACTION_GENERIC', 'NEG_ACTION_OCCURRENCE',
    'NEG_ACTION_PERCEPTION', 'NEG_ACTION_REPORTING', 'NEG_ACTION_STATE',
        'TIME_DATE', 'TIME_DURATION', 'TIME_OF_THE_DAY', 'TIME_REPETITION',
        'LOC_FAC', 'LOC_GEO', 'LOC_OTHER',
        'HUMAN_PART', 'HUMAN_PART_FAC', 'HUMAN_PART_GENERIC', 'HUMAN_PART_GPE', 'HUMAN_PART_MET', 'HUMAN_PART_ORG',
    'HUMAN_PART_PER', 'HUMAN_PART_VEH',
        'NON_HUMAN_PART', 'NON_HUMAN_PART_GENERIC',
        'UNKNOWN_INSTANCE_TAG'
        'TIME',
        'LOC',
        'HUMAN',
        'NON_HUMAN',
        'ACTION',
        'NEG_ACTION',
        ('ACTION', 'NEG_ACTION',),
        ('HUMAN', 'NON_HUMAN',),
        ('TIME', 'LOC', 'HUMAN', 'NON_HUMAN',)
    }
    # 计算每个句子中Argument到trigger的距离
    G = nx.Graph()
    for trigger in (m1, m2):
        # 计算argument到trigger的最短距离。注意，argument是EcbInstance不是EcbComponent，但trigger都是EcbComponent
        argument_distance_dict = defaultdict(lambda: 10000)
        G.add_node(trigger)
        for component in trigger.sentence.components_dict.values():
            if component.tag.startswith(("ACTION", "NEG_ACTION", 'UNKNOWN')):
                continue
            # 计算距离
            argument = component.instance_global if cross_document else component.instance_within
            argument_distance_dict[argument] = min(argument_distance_dict[argument],
                                                   trigger_argument_distance(trigger, component))
            # 计算feature
            feature_dict = dict()
            for tag in tags:
                feature_dict[tag] = argument.tag.startswith(tag)
        # 添加边
        for argument, distance in argument_distance_dict.items():
            G.add_edge(trigger, argument, weight=normalize_edge_weight(distance))
    # 添加每个点的feature
    feature_for_each_node = dict()
    for node in G.nodes:
        feature_dict = dict()
        for tag in tags:
            feature_dict[tag] = int(node.tag.startswith(tag))
        feature_for_each_node[node] = [feature_dict[tag] for tag in tags]  # 保证每个点feature顺序一致
    nx.set_node_attributes(G, name="feature", values=feature_for_each_node)  # 批量为每个点添加feature
    return G


def convert_mention_pair_to_GNN_features(mention_pair: MentionPair, cross_document: bool):
    G = gnn_graph(m1=mention_pair.mention_pair[0], m2=mention_pair.mention_pair[1], cross_document=cross_document)
    # GNN feature
    node_id = dict()
    for index, node in enumerate(G.nodes):
        node_id[node] = index
    edges = []
    weights = []
    for f, t in G.edges:
        edges.append([node_id[f], node_id[t]])
        edges.append([node_id[t], node_id[f]])
        weights.append(G[f][t]["weight"])
        weights.append(G[f][t]["weight"])

    x = torch.tensor([G.node[node]["feature"] for node in G.nodes], dtype=torch.float)
    try:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    except RuntimeError:
        edge_index = torch.tensor([[],[]], dtype=torch.long)  # 没有边时，放一个空张量
    edge_weight = torch.tensor(weights, dtype=torch.float)
    y = torch.tensor([mention_pair.label(cross_document=cross_document)], dtype=torch.long)
    trigger_mask = torch.tensor([[int(isinstance(node, EcbComponent))] for node in G.nodes], dtype=torch.long)
    data = GnnData(x=x, edge_index=edge_index, y=y)
    data.edge_weight = edge_weight
    data.trigger_mask = trigger_mask
    return data


def draw_argument_graph(G: nx.Graph):
    node_labels = {}
    trigger_nodes, argument_nodes = [], []
    for node in G.nodes:
        if isinstance(node, EcbComponent):  # trigger
            node_labels[node] = node.text
            trigger_nodes.append(node)
        elif isinstance(node, EcbInstance):  # argument
            node_labels[node] = node.mentions_list[0].text
            argument_nodes.append(node)
    limits = plt.axis('off')  # 不显示坐标
    layout = nx.spring_layout(G)  # 生成图的布局
    nx.draw_networkx_nodes(G, layout, nodelist=trigger_nodes, node_color="b", node_size=900)  # 绘制两个trigger的点，画成蓝色蓝色
    nx.draw_networkx_nodes(G, layout, nodelist=argument_nodes, node_color="r") # 画出其他argument点，画成红色
    nx.draw_networkx_labels(G, layout, labels=node_labels)  # 绘制点的标签
    for f, t in G.edges:  # 绘制边。每次画一个边，用粗细表示权重
        nx.draw_networkx_edges(G, layout, edgelist=[(f, t)], width=G[f][t]["weight"] * 10)
    edge_labels = {(f, t): round(G[f][t]["weight"], 2) for f, t in G.edges}
    # nx.draw_networkx_edge_labels(G, layout, edge_labels=edge_labels)  # 绘制边的标签
    plt.show()
    plt.clf()
    limits = plt.axis('on')