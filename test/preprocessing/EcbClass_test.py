#%%
from collections import defaultdict
from preprocessing.Structurize.EcbClass import *
from pandas import DataFrame, Series
#%%
ecb = EcbPlusTopView()


#%% 聚类簇是否正确
instance_global: EcbInstance = ecb.coreference_view.instance_dict_by_topic[1]['ACT16236402809085484']
# 一个全局聚类簇下的component的instance_global全部指向该全局簇
assert all([component.instance_global == instance_global for component in instance_global.mentions_list])
# 没有把全局簇和内部簇弄混
assert all([component.instance_within != instance_global for component in instance_global.mentions_list])
# 一个内部聚类簇下的component的instance_global全部指向该内部簇
instance_within: EcbInstance = ecb.document_view.topics_dict[1].documents_dict['1_1ecb.xml'].instances_dict['HUM16236184328979740']
assert all([component.instance_within == instance_within for component in instance_within.mentions_list])

#%%
# 所有argument的tag种类
tag_list = []
for topic in ecb.coreference_view.instance_dict_by_topic.values():
    for iid, instance in topic.items():
        instance: EcbInstance = instance
        tag_list.append(instance.tag)
tag_set = set(tag_list)
tags = {
    'ACTION_ASPECTUAL', 'ACTION_CAUSATIVE', 'ACTION_GENERIC', 'ACTION_OCCURRENCE', 'ACTION_PERCEPTION', 'ACTION_REPORTING', 'ACTION_STATE',
    'NEG_ACTION_ASPECTUAL', 'NEG_ACTION_CAUSATIVE', 'NEG_ACTION_GENERIC', 'NEG_ACTION_OCCURRENCE', 'NEG_ACTION_PERCEPTION', 'NEG_ACTION_REPORTING', 'NEG_ACTION_STATE',
    'TIME_DATE', 'TIME_DURATION', 'TIME_OF_THE_DAY', 'TIME_REPETITION',
    'LOC_FAC', 'LOC_GEO', 'LOC_OTHER',
    'HUMAN_PART', 'HUMAN_PART_FAC', 'HUMAN_PART_GENERIC', 'HUMAN_PART_GPE', 'HUMAN_PART_MET', 'HUMAN_PART_ORG', 'HUMAN_PART_PER', 'HUMAN_PART_VEH',
    'NON_HUMAN_PART', 'NON_HUMAN_PART_GENERIC',
    'UNKNOWN_INSTANCE_TAG'}
assert tag_set == tags

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
