import sys
sys.path.append("../..")

from itertools import combinations, product, chain
from tqdm import tqdm
import numpy as np
from pandas import DataFrame, Series, read_csv
import csv
import networkx as nx
from pytorch_pretrained_bert.tokenization import BertTokenizer

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
            input_ids,
            input_mask,
            segment_ids,
            trigger_mask_a,
            trigger_mask_b,
            tf_idf_a,
            tf_idf_b,
            label
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
        self.TF_IDF = [tf_idf_a, tf_idf_b]
        # 图网络特征
        # todo


class InputFeaturesCreator(object):
    def __init__(self, ECBPlus: EcbPlusTopView):
        self.ecb = ECBPlus

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
            feature = InputFeatures(
                id_a=pair.mention_pair[0].global_mid(),
                id_b=pair.mention_pair[1].global_mid(),
                label=int(pair.label(cross_document=cross_document)),
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                trigger_mask_a=trigger_mask_a,
                trigger_mask_b=trigger_mask_b,
                tf_idf_a=[0,1], # todo
                tf_idf_b=[1,0]
            )
            features.append(feature)
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

# class MentionPairCreator(object):
#     def __init__(self, ECBPLUS: EcbPlusTopView):
#         self.ECBPLUS: EcbPlusTopView = ECBPLUS
#
#     def generate_mention_pairs(
#             self,
#             topics: list or str="all",
#             cross_document: bool=True,
#             cross_topic: bool=True,
#             prefix_list: list=None,
#             ignore_order: bool=True,
#             positive_increase: int=0,
#             shuffle: bool=True,
#             by_what: str="mention"):
#         topics_list = list(self.ECBPLUS.document_view.topics_dict.keys()) if topics == "all" else topics[:]
#         mention_pair_dict = dict()
#
#         # 用来添加mention-pair 的函数
#         def add_mention_pair(m1, m2, mention_pair_dict, ignore_order: bool = True, by_what: str="mention"):
#             if by_what=="sentence" and m1.sentence.text() == m2.sentence.text():
#                 return
#             mention_pair = MentionPair(mention1=m1, mention2=m2, ignore_order=ignore_order)
#             if mention_pair.pair_id(by_what=by_what) in mention_pair_dict.keys():
#                 exist_pair = mention_pair_dict[mention_pair.pair_id(by_what=by_what)]
#                 if exist_pair.fixed_label in (None, 0) and mention_pair.label(cross_document=cross_document) ==1:
#                     exist_pair.fixed_label = 1
#             else:
#                 mention_pair_dict[mention_pair.pair_id(by_what=by_what)] = mention_pair
#                 mention_pair.fixed_label = mention_pair.label(cross_document=cross_document)
#
#         mentions_all = []  # 所有mention
#         mentions_by_topic = {}  # 每个topic中的mention
#         mentions_by_document = {}  # 每个文档中的mention
#         # 搜出所有mention
#         for topic_id in topics_list:
#             try:
#                 topic: EcbTopic = self.ECBPLUS.document_view.topics_dict[topic_id]
#             except KeyError:
#                 print("no topic %s" % topic_id)
#                 continue
#             mentions_by_topic[topic_id] = []
#             for document in topic.documents_dict.values():
#                 document: EcbDocument = document
#                 mentions_by_document[document.document_name] = []
#                 # 筛选文档中的mention
#                 if prefix_list:
#                     mentions_in_doc = [comp for comp in document.components_dict.values() if
#                                        comp.tag.startswith(tuple(prefix_list))]
#                 else:
#                     mentions_in_doc = [comp for comp in document.components_dict.values()]
#                 mentions_by_document[document.document_name] = mentions_in_doc
#                 mentions_by_topic[topic_id] += mentions_in_doc
#                 mentions_all += mentions_in_doc
#
#         # 生成mention pair
#         if cross_topic:  # 跨document，跨topic
#             for m1, m2 in tqdm(product(mentions_all, repeat=2), desc="Generate mention-pair [cross topic]:"):
#                 add_mention_pair(m1, m2, mention_pair_dict, ignore_order=ignore_order, by_what=by_what)
#         elif cross_document:  # 跨document，不跨topic
#             for mentions_in_topic in tqdm(mentions_by_topic.values(), desc="Generate mention-pair [cross document]:"):
#                 for m1, m2 in product(mentions_in_topic, repeat=2):
#                     add_mention_pair(m1, m2, mention_pair_dict, ignore_order=ignore_order, by_what=by_what)
#         else:  # document内
#             for mentions_in_doc in tqdm(mentions_by_document.values(), desc="Generate mention-pair [within document]:"):
#                 for m1, m2 in product(mentions_in_doc, repeat=2):
#                     add_mention_pair(m1, m2, mention_pair_dict, ignore_order=ignore_order, by_what=by_what)
#         result = list(mention_pair_dict.values())
#
#         # 扩增正例
#         if positive_increase > 0:
#             positive_samples = [pair for pair in result if pair.label(cross_document=cross_document)]
#             result += positive_samples * positive_increase
#
#         result = np.array(result)
#         # shuffle
#         if shuffle:
#             np.random.shuffle(result)
#         return result
#
#     def BERT_MRPC_feature(
#             self,
#             topics: str = "all",
#             cross_document: bool = True,
#             cross_topic: bool = True,
#             positive_increase: int=0,
#             shuffle: bool=True,
#             csv_path: str="",
#             to_csv: bool=True) -> DataFrame:
#         mention_pair_array: np.array = self.generate_mention_pairs(
#             topics=topics, cross_document=cross_document, cross_topic=cross_topic, prefix_list=["ACTION", "NEG_ACTION"],
#             ignore_order=True, positive_increase=positive_increase, shuffle=shuffle, by_what="sentence")
#         # table = DataFrame(columns=["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"])
#         # for pair in mention_pair_array:
#         #     pair: MentionPair = pair
#         #     if table.empty \
#         #             or pair.pair_id(by_what="sentence") not in table.index \
#         #             or table.loc[pair.pair_id(by_what="sentence"), "Quality"] == 0 and pair.label(cross_document=cross_document)==1:
#         #         new_data = Series({
#         #             "Quality": pair.label(cross_document=cross_document),
#         #             "#1 ID": pair.component_pair[0].sentence.sid(),
#         #             "#2 ID": pair.component_pair[1].sentence.sid(),
#         #             "#1 String": pair.component_pair[0].sentence.text(),
#         #             "#2 String": pair.component_pair[1].sentence.text(),
#         #         })
#         #         table.loc[pair.pair_id(by_what="sentence"), :] = new_data
#         table = DataFrame()
#         table["Quality"] = Series([pair.fixed_label for pair in mention_pair_array])
#         table["#1 ID"] = Series([pair.component_pair[0].sentence.sid() for pair in mention_pair_array])
#         table["#2 ID"] = Series([pair.component_pair[1].sentence.sid() for pair in mention_pair_array])
#         table["#1 String"] = Series([pair.component_pair[0].sentence.text() for pair in mention_pair_array])
#         table["#2 String"] = Series([pair.component_pair[1].sentence.text() for pair in mention_pair_array])
#         if to_csv:
#             table.to_csv(csv_path, sep="\t", index=False)
#         return table
#
#     def generate_sentence_pair(
#             self,
#             topics: list or str="all",
#             selected: bool = True,
#             cross_document: bool=True,
#             cross_topic: bool=True,
#             ignore_order: bool=True):
#
#         topics_list = list(self.ECBPLUS.document_view.topics_dict.keys()) if topics == "all" else topics[:]
#         sentences_all = []
#         sentences_by_topic = dict()
#         sentences_by_document = dict()
#         # 搜出所有sentence
#         for topic_id in topics_list:
#             try:
#                 topic: EcbTopic = self.ECBPLUS.document_view.topics_dict[topic_id]
#             except KeyError:
#                 print("no topic %s" % topic_id)
#                 continue
#             sentences_by_topic[topic_id] = []
#             for document in topic.documents_dict.values():
#                 document: EcbDocument = document
#                 if selected:
#                     sentences_in_doc = [s for s in document.all_sentences_dict.values() if s.selected]
#                 else:
#                     sentences_in_doc = [s for s in document.all_sentences_dict.values()]
#                 sentences_by_document[document.document_name] = sentences_in_doc
#                 sentences_by_topic[topic_id] += sentences_in_doc
#                 sentences_all += sentences_in_doc
#
#         sentences_pairs = []
#         if cross_topic:  # 跨document，跨topic
#             pairs = combinations(sentences_all, r=2) if ignore_order else product(sentences_all, repeat=2)
#             for s1, s2 in tqdm(pairs, desc="Generate sentence-pair [cross topic]:"):
#                 sentences_pairs.append((s1, s2))
#         elif cross_document:  # 跨document，不跨topic
#             for mentions_in_topic in tqdm(sentences_by_topic.values(), desc="Generate sentence-pair [cross document]:"):
#                 pairs = combinations(mentions_in_topic, r=2) if ignore_order else product(mentions_in_topic, repeat=2)
#                 for s1, s2 in pairs:
#                     sentences_pairs.append((s1, s2))
#         else:  # document内
#             for mentions_in_doc in tqdm(sentences_by_document.values(), desc="Generate sentence-pair [within document]:"):
#                 pairs = combinations(mentions_in_doc, r=2) if ignore_order else product(mentions_in_doc, repeat=2)
#                 for s1, s2 in pairs:
#                     sentences_pairs.append((s1, s2))
#         return sentences_pairs
#
#     def argument_feature(
#             self,
#             topics: str = "all",
#             cross_document: bool = True,
#             cross_topic: bool = True,
#             positive_increase: int=0,
#             shuffle: bool=True,
#             csv_path: str="",
#             to_csv: bool=True) -> DataFrame:
#         sentences_pairs = self.generate_sentence_pair(
#             topics=topics, selected=True, cross_document=cross_document, cross_topic=cross_topic, ignore_order=True)
#         sentences_set = set(chain.from_iterable(sentences_pairs))
#         sentences_instances = dict()
#         # 生成每个句子的每类Argument集合
#         for sentence in tqdm(sentences_set, desc="collecting arguments"):
#             sentence: EcbSentence = sentence
#             sentences_instances[sentence] = defaultdict(set)
#             # 添加每个子类component的集合
#             for component in sentence.components_dict.values():
#                 component: EcbComponent = component
#                 if component.tag.startswith("'UNKNOWN"):
#                     continue
#                 instance = component.instance_global if cross_document else component.instance_within
#                 sentences_instances[sentence][component.tag].add(instance)
#             # 添加time，loc，human_participate,non_human_prticipate,action,neg_action的集合
#             actions = set()
#             neg_actions = set()
#             times = set()
#             locations = set()
#             human_participates = set()
#             non_human_participates = set()
#             for sub_tag, instances_set in sentences_instances[sentence].items():
#                 if sub_tag.startswith("ACTION"):
#                     actions.update(instances_set)
#                 elif sub_tag.startswith("NEG_ACTION"):
#                     neg_actions.update(instances_set)
#                 elif sub_tag.startswith("TIME"):
#                     times.update(instances_set)
#                 elif sub_tag.startswith("LOC"):
#                     locations.update(instances_set)
#                 elif sub_tag.startswith("HUMAN"):
#                     human_participates.update(instances_set)
#                 elif sub_tag.startswith("NON_HUMAN"):
#                     non_human_participates.update(instances_set)
#                 else:
#                     raise KeyError(f"{sub_tag} is an illegal component tag")
#             sentences_instances[sentence]["ACTION"].update(actions)
#             sentences_instances[sentence]["NEG_ACTION"].update(neg_actions)
#             sentences_instances[sentence]["TIME"].update(times)
#             sentences_instances[sentence]["LOC"].update(locations)
#             sentences_instances[sentence]["HUMAN"].update(human_participates)
#             sentences_instances[sentence]["NON_HUMAN"].update(non_human_participates)
#
#             sentences_instances[sentence]["ACTION_and_NEG_ACTION"].update(actions | neg_actions)
#             sentences_instances[sentence]["HUMAN_and_NON_HUMAN"].update(human_participates | non_human_participates)
#             sentences_instances[sentence]["Arguments"].update(times | locations | human_participates | non_human_participates)
#
#         df_arg_feature = DataFrame(columns=["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"])
#         for s1, s2 in tqdm(sentences_pairs, desc="generate records"):
#             s1: EcbSentence = s1
#             s2: EcbSentence = s2
#             if s1.text() == s2.text():
#                 continue
#             record = Series({"#1 ID":s1.sid(), "#2 ID":s2.sid(), "#1 String":s1.text(), "#2 String": s2.text()})
#             record["Quality"] = int((sentences_instances[s1]["ACTION_and_NEG_ACTION"] & sentences_instances[s2]["ACTION_and_NEG_ACTION"]) != set())
#             df_arg_feature = df_arg_feature.append(record, ignore_index=True)
#         return df_arg_feature