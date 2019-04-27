#%%
from collections import defaultdict
from preprocessing.Structurize.EcbClass import *
from preprocessing.Feature.MentionPair import MentionPairCreator, InputFeaturesCreator
from pandas import DataFrame, Series

#%%
ecb = EcbPlusTopView()
IFC = InputFeaturesCreator(ecb)

topics = range(1,35)
cross_document = False
max_seq_len = 123
trigger_half_window = 2

examples_within = IFC.create_from_dataset(
    topics=topics, cross_document=False, cross_topic=False,
    max_seq_length=max_seq_len, trigger_half_window=trigger_half_window)

examples_cross = IFC.create_from_dataset(
    topics=topics, cross_document=True, cross_topic=False,
    max_seq_length=max_seq_len, trigger_half_window=trigger_half_window)

examples_global = IFC.create_from_dataset(
    topics=topics, cross_document=True, cross_topic=True,
    max_seq_length=max_seq_len, trigger_half_window=trigger_half_window)

