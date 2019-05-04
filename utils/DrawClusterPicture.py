import os
from collections import deque
from pandas import Series, DataFrame, Index
import matplotlib.pyplot as plt
import numpy as np
import json
import networkx as nx
from random import random as rand
import pydot
from networkx.drawing.nx_pydot import graphviz_layout


def draw_cluster_graph(example: list, pred, figsize=50, layout_func=nx.spring_layout):
    id_a = [ex.id_a for ex in example]
    id_b = [ex.id_b for ex in example]
    turth = [int(ex.label) for ex in example]
    pred = [int(p) for p in pred]
    # 生成共指图
    G_turth, G_pred = nx.Graph(), nx.Graph()
    id_set = set(id_a+id_b)
    G_turth.add_nodes_from(id_set)
    G_pred.add_nodes_from(id_set)
    for label_list, G in zip([turth, pred], [G_turth, G_pred]):
        for a,b,label in zip(id_a, id_b, label_list):
            if label==1:
                G.add_edge(a,b)
    # 生成共指簇
    clusters_turth = nx.connected_components(G_turth)

    # 画图
    for G, name in [(G_pred, "pred"), (G_turth, "turth"),]:
        limits = plt.axis('off')  # 不显示坐标
        plt.figure(figsize=(figsize, figsize))
        # layout = nx.spring_layout(G)
        layout = layout_func(G)
        # layout = graphviz_layout(G)
        singletons = []
        non_singletons = []
        for index, cluster_nodes in enumerate(clusters_turth, start=1):
            if len(cluster_nodes) == 1:
                singletons.append(list(cluster_nodes)[0])
            else:
                non_singletons.append(cluster_nodes)

        singleton_size = 10
        nx.draw_networkx_nodes(G_turth, layout, nodelist=singletons, node_size=singleton_size, node_color="black")

        for index, cluster_nodes in enumerate(non_singletons, start=1):
            color = [(rand(), rand(), rand())] * len(cluster_nodes)
            cmap = plt.cm.gist_rainbow
            node_size = 20
            nx.draw_networkx_nodes(G_turth, layout, nodelist=cluster_nodes, node_size=node_size, node_color=color, cmap=cmap)

        nx.draw_networkx_edges(G, layout, edgelist=G.edges)
        # plt.show()
        plt.savefig("./%s_cluster_image.png" % name)
        plt.clf()
        # limits = plt.axis('on')

