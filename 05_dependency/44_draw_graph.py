"""
Visualize a dependency tree of a sentence as a directed graph. 
Consider converting a dependency tree into DOT language and use Graphviz for drawing a directed graph. 
In addition, you can use pydot for drawing a dependency tree.
https://nlp100.github.io/en/ch05.html#44-visualize-dependency-trees
"""

import load_data
from chunk_sentence import Sentece

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
# needed for displaying Japanese character
import japanize_matplotlib


matplotlib.use('Svg')

root = load_data.load()
first_sentence = root[2]
sentece = Sentece(first_sentence)


G = nx.DiGraph()

for chunk in sentece.chunks:
    G.add_node(chunk.id, text=chunk.to_text())

for chunk in sentece.chunks:
    if chunk.dst == -1:
        continue
    G.add_edge(chunk.id, chunk.dst)

# for debugging
# print(list(G.edges(data=True)))
# print(list(G.nodes(data=True)))

# specify the algorithm that calculate positions of nodes
# https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
pos = nx.spring_layout(G, k=2)

# set font_size 0 to remove node id
nx.draw_networkx(G, pos, font_size=0, node_size=50,
                 margins=[0.05, 0.05], node_color="#E0FFFF", edge_color="#A9A9A9")

# display labels
node_labels = nx.get_node_attributes(G, 'text')
nx.draw_networkx_labels(G, pos, labels=node_labels,
                        font_family='IPAexGothic')
plt.tight_layout()
plt.savefig("result_44_draw_graph.svg")

# question:
# I want to prevent nodes from appearing over another node. But it is quite difficult for me
# Is there something useful method?
