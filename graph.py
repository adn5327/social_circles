import community
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time

import setup_matrix


def setup_graph(filename):
    adj = setup_matrix.read_input(filename)
    G = nx.from_numpy_matrix(adj)
    return G

def detect_communities(G):
    best = community.best_partition(G)
    return best

def run(filename):
    print('graph setup')
    G = setup_graph(filename)
    print('about to detect communities')
    partition = detect_communities(G)

    #drawing
    size = float(len(set(partition.values())))
    print('spring layout')
    pos = nx.spring_layout(G)
    count = 0.
    print('starting to draw graph')
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                            if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                            node_color = str(count / size))
    nx.draw_networkx_edges(G,pos, alpha=0.5)
    plt.savefig('out{}.png'.format(int(time.time())))
    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        run('data/facebook_combined.txt')
    else:
        run(sys.argv[1])
