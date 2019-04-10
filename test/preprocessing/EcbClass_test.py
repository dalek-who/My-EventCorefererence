#%%
from collections import defaultdict
from preprocessing.Structurize.EcbClass import *
from pandas import DataFrame, Series
#%%
ecb = EcbPlusTopView()

#%%
# 统计每篇文章的句子长度
sentence_token_num = defaultdict(lambda :0)
for topic_id, ecb_topic in edv.topics_dict.items():
    topic_id: int
    ecb_topic: EcbTopic
    for doc_name, ecb_doc in ecb_topic.documents_dict.items():
        ecb_doc: EcbDocument
        for sid, sentence in ecb_doc.all_sentences_dict.items():
            sentence: EcbSentence
            sentence_token_num[len(sentence.tokens_list)] += 1
#%%
sr_sentence_token_num = Series(dict(sentence_token_num))
sr_sentence_token_num.sort_index(inplace=True)
