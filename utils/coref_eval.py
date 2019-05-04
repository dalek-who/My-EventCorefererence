import networkx as nx
from neleval.coref_metrics import b_cubed, muc, mention_ceaf, entity_ceaf, pairwise, pairwise_negative, _prf

def evaluate_coref_from_example(example: list, pred):
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
    clusters_turth, clusters_pred = dict(), dict()
    for clusters, G in zip([clusters_turth, clusters_pred], [G_turth, G_pred]):
        for cid, c in enumerate(nx.connected_components(G)):
            clusters["NIL%s" % cid] = set(c)

    # 计算coref指标
    coref_eval = dict()
    for metrics_name, metrics_func in zip(["MUC", "B-cubed", "CEAFe", "CEAFm", "BLANCc", "BLANCn"],
                                          [b_cubed, muc, entity_ceaf, mention_ceaf, pairwise, pairwise_negative]):
        p, r, f1 = _prf(*metrics_func(clusters_turth, clusters_pred))
        coref_eval[metrics_name + "_p"] = p
        coref_eval[metrics_name + "_r"] = r
        coref_eval[metrics_name + "_f1"] = f1
    coref_eval["BLANC_p"] = (coref_eval["BLANCc_p"] + coref_eval["BLANCn_p"]) / 2
    coref_eval["BLANC_r"] = (coref_eval["BLANCc_r"] + coref_eval["BLANCn_r"]) / 2
    coref_eval["BLANC_f1"] = (coref_eval["BLANCc_f1"] + coref_eval["BLANCn_f1"]) / 2

    coref_eval["CoNLL_p"] = (coref_eval["B-cubed_p"] + coref_eval["MUC_p"] + coref_eval["CEAFe_p"]) / 3
    coref_eval["CoNLL_r"] = (coref_eval["B-cubed_r"] + coref_eval["MUC_r"] + coref_eval["CEAFe_r"]) / 3
    coref_eval["CoNLL_f1"] = (coref_eval["B-cubed_f1"] + coref_eval["MUC_f1"] + coref_eval["CEAFe_f1"]) / 3

    coref_eval["AVG_p"] = (coref_eval["B-cubed_p"] + coref_eval["MUC_p"] + coref_eval["CEAFe_p"] + coref_eval["BLANC_p"]) / 4
    coref_eval["AVG_r"] = (coref_eval["B-cubed_r"] + coref_eval["MUC_r"] + coref_eval["CEAFe_r"] + coref_eval["BLANC_r"]) / 4
    coref_eval["AVG_f1"] = (coref_eval["B-cubed_f1"] + coref_eval["MUC_f1"] + coref_eval["CEAFe_f1"] + coref_eval["BLANC_f1"]) / 4

    return coref_eval

