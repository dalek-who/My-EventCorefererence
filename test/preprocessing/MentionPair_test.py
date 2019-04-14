#%%
from collections import defaultdict
from preprocessing.Structurize.EcbClass import *
from preprocessing.Feature.MentionPair import MentionPairCreator
from pandas import DataFrame, Series

#%%
ecb = EcbPlusTopView()
mpg = MentionPairCreator(ecb)
#%%
topics = "all"
mention_pairs_global = mpg.generate_mention_pairs(topics=topics, cross_document=True, prefix_list=["ACTION", "NEG_ACTION"], ignore_order=True)
mention_pairs_within = mpg.generate_mention_pairs(topics=topics, cross_document=False, prefix_list=["ACTION", "NEG_ACTION"], ignore_order=True)
#%%
assert len(mention_pairs_global) > len(mention_pairs_within)
assert len([pair for pair in mention_pairs_global if pair.label(cross_document=True)]) == 53473
assert len(mention_pairs_global) == 3171863
assert len([pair for pair in mention_pairs_within if pair.label(cross_document=False)]) == 4071
assert len(mention_pairs_within) == 195439
#%%
def positive_rate(pairs_list, cross_document):
    pos_num = len([pair for pair in pairs_list if pair.label(cross_document=cross_document)])
    total_num = len(pairs_list)
    return pos_num / total_num

increace = 3
topics = range(1,11)
pos_inc = mpg.generate_mention_pairs(topics=topics, cross_document=True, prefix_list=["ACTION", "NEG_ACTION"], ignore_order=True, positive_increase=increace)
print(positive_rate(pos_inc, cross_document=True))
