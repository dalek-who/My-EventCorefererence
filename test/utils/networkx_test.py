#%%
import networkx as nx
#%%
NodeIds = ["A", "B", "C", "D", 'E', 'F', 'G', 'H', 'I', 'J', 'K']
Links = [("A",'B'), ('A','C'), ('A','D'),
         ('E','F'),('F','G'),('G','H'),('H','I')]

#%%
G = nx.Graph()
G.add_nodes_from(NodeIds)
G.add_edges_from(Links)

#%%
Instances = [list(c) for c in nx.connected_components(G)]