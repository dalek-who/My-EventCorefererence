#%%
import os
from preprocessing.Structurize.EcbClass import EcbPlusTopView
from preprocessing.Feature.MentionPair import MentionPairCreator
#%%
train_topics = list(range(1, 36))
dev_topics = [12, 18, 21, 23, 34, 35]
cross_document = True
cross_topic=False
to_csv = False
#%%
ECB = EcbPlusTopView()
MPC = MentionPairCreator(ECB)
#%%
arg_feature = MPC.argument_feature(
    topics=train_topics, cross_document=cross_document, cross_topic=cross_topic, positive_increase=0,
    shuffle=False, csv_path="", to_csv=to_csv)
#%%
arg_feature_sorted = arg_feature.sort_values(by=["#1 String", "#2 String"])
#%%
arg_feature_cross_topic = MPC.argument_feature(
    topics=train_topics, cross_document=True, cross_topic=True, positive_increase=0,
    shuffle=False, csv_path="", to_csv=to_csv)
