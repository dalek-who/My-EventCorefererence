from preprocessing.Structurize.EcbClass import *
from preprocessing.Feature.MentionPair import MentionPairCreator, InputFeaturesCreator
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data as GnnData



#%%
ecb = EcbPlusTopView()
IFC = InputFeaturesCreator(ecb)

topics = range(1,35)
max_seq_len = 123
trigger_half_window = 2
cross_document = False
#%%
pairs_within = MentionPairCreator(ecb).generate_mention_pairs(topics="all", cross_document=cross_document, cross_topic=False,
                                                                 prefix_list=["ACTION","NEG_ACTION"], ignore_order=True)

#%%
example_mention_pair = None
for pair in pairs_within:
    if len(pair.mention_pair[0].sentence.components_dict) >3 \
        and len(pair.mention_pair[1].sentence.components_dict)>3 \
        and pair.label():
        example_mention_pair = pair
        break
assert example_mention_pair is not None

#%%
m1 = example_mention_pair.mention_pair[0]
m2 = example_mention_pair.mention_pair[1]
print(m1.sentence.text_with_annotation())
print(m2.sentence.text_with_annotation())


#%%
from collections import defaultdict
from math import log
import matplotlib.pyplot as plt


def trigger_argument_distance(trigger: EcbComponent, argument: EcbComponent):
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

#%%

tags = {
    'ACTION_ASPECTUAL', 'ACTION_CAUSATIVE', 'ACTION_GENERIC', 'ACTION_OCCURRENCE', 'ACTION_PERCEPTION', 'ACTION_REPORTING', 'ACTION_STATE',
    'NEG_ACTION_ASPECTUAL', 'NEG_ACTION_CAUSATIVE', 'NEG_ACTION_GENERIC', 'NEG_ACTION_OCCURRENCE', 'NEG_ACTION_PERCEPTION', 'NEG_ACTION_REPORTING', 'NEG_ACTION_STATE',
    'TIME_DATE', 'TIME_DURATION', 'TIME_OF_THE_DAY', 'TIME_REPETITION',
    'LOC_FAC', 'LOC_GEO', 'LOC_OTHER',
    'HUMAN_PART', 'HUMAN_PART_FAC', 'HUMAN_PART_GENERIC', 'HUMAN_PART_GPE', 'HUMAN_PART_MET', 'HUMAN_PART_ORG', 'HUMAN_PART_PER', 'HUMAN_PART_VEH',
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
    for component in trigger.sentence.components_dict.values():
        if component.tag.startswith(("ACTION", "NEG_ACTION",'UNKNOWN')):
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

# GNN feature
node_id = dict()
for index, node in enumerate(G.nodes):
    node_id[node] = index
edges = []
weights = []
for f,t in G.edges:
    edges.append([node_id[f], node_id[t]])
    edges.append([node_id[t], node_id[f]])
    weights.append(G[f][t]["weight"])
    weights.append(G[f][t]["weight"])

x = torch.tensor([G.node[node]["feature"] for node in G.nodes], dtype=torch.long)
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(weights, dtype=torch.float)
y = torch.tensor([pair.label(cross_document=cross_document)], dtype=torch.long)
trigger_mask = torch.tensor([[int(isinstance(node, EcbComponent))] for node in G.nodes], dtype=torch.long)
data = GnnData(x=x, edge_index=edge_index, y=y)
data.edge_weight = edge_weight
data.trigger_mask = trigger_mask

#%%
ft = np.array([G.node[node]["feature"] for node in G.nodes])
#%%
labels = {}
for node in G.nodes:
    if isinstance(node, EcbComponent): # trigger
        labels[node] = node.text
    elif isinstance(node, EcbInstance):  # argument
        labels[node] = node.mentions_list[0].text
nx.draw_networkx(G,nx.spring_layout(G), labels=labels)
plt.show()

#%%
# Argument-Trigger图
limits=plt.axis('off')  # 不显示坐标
layout = nx.spring_layout(G)  # 生成图的布局
nx.draw_networkx_nodes(G,layout, nodelist=[m1,m2], node_color="b", node_size=900)  # 绘制两个trigger的点，画成蓝色蓝色
argument_nodes = [node for node in G.nodes if isinstance(node, EcbInstance)]  # 画出其他argument点，画成红色
nx.draw_networkx_nodes(G, layout, nodelist=argument_nodes, node_color="r")
nx.draw_networkx_labels(G, layout, labels=labels)  # 绘制点的标签
for f,t in G.edges:  # 绘制边。每次画一个边，用粗细表示权重
    nx.draw_networkx_edges(G, layout, edgelist=[(f,t)], width=G[f][t]["weight"]*10)
edge_labels = {(f,t): round(G[f][t]["weight"], 2) for f,t in G.edges}
# nx.draw_networkx_edge_labels(G, layout, edge_labels=edge_labels)  # 绘制边的标签
plt.show()

#%%
from pandas import Series
def edge_weight_n(distance: int, n=1):
    return log(1 + n/(0.5 + distance))
Series([edge_weight_n(i, n=15) for i in range(15)]).plot()
plt.show()