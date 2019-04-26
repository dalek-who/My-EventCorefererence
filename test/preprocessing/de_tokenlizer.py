""" 用tokenlizer分解为WordPiece后，为了做trigger embedding，还需要查到每个token对应的 word-piece"""
#%%
def find_word_piece_index_for_tokens(tokens: list, word_pieces: list):
    i, j = 0, 0
    pairs = []
    start_a_new_token = True
    while i < len(tokens) and j < len(word_pieces):
        current_token = tokens[i].lower().rstrip("\t").replace("�","").replace("ô","o") # 数据集中这些符号在tokenlize时会变得和原来不一样，直接去掉
        if start_a_new_token:
            new_token = ""
            new_pair = [tokens[i], i, [], []]
        new_piece = word_pieces[j] if not word_pieces[j].startswith("##") else word_pieces[j][2:]
        new_token += new_piece
        if new_token == current_token:
            new_pair[2].append(word_pieces[j])
            new_pair[3].append(j)
            pairs.append(tuple(new_pair))
            i += 1
            start_a_new_token = True
        else:
            new_pair[2].append(new_piece)
            new_pair[3].append(j)
            start_a_new_token = False
        j += 1
    is_success = i == len(tokens) and j == len(word_pieces) and len(pairs)==len(tokens)
    return pairs, is_success, i, j


#%%
# 对ECB+数据集的一些统计
import sys
sys.path.append("../..")
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from pytorch_pretrained_bert.tokenization import BertTokenizer
from preprocessing.Structurize.EcbClass import *
from tqdm import tqdm

#%%
ECB = EcbPlusTopView()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

#%%
failed = []
for topic_id, topic in tqdm(ECB.document_view.topics_dict.items()):
    for doc_name, doc in topic.documents_dict.items():
        for sid, sentence in doc.all_sentences_dict.items():
            if sentence.selected:
                sentence: EcbSentence = sentence
                tokens = [t.word for t in sentence.tokens_list]
                word_pieces = tokenizer.tokenize(sentence.text())
                pairs, is_success, i, j = find_word_piece_index_for_tokens(tokens, word_pieces)
                if not is_success:
                    new_failed = (list(enumerate(tokens)), list(enumerate(word_pieces)), pairs, sentence.sid())
                    failed.append(new_failed)
if len(failed)==0:
    print("success")