#%%
import pickle
from preprocessing.Structurize.EcbClass import EcbPlusTopView
from preprocessing.Feature.MentionPair import InputFeaturesCreator
import numpy as np
#%%
ecb = EcbPlusTopView()
#%%
ecb.dump()
#%%
load_ecb = EcbPlusTopView.load()