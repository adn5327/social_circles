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

def getTrue(filename):
    true = []
    with open(filename, 'r') as f:
        for line in f:
            circle = []
            x = line.rstrip().split()
            for node in x[1:]:
                circle.append(int(node))
            true.append(np.array(circle))

def evaluate_score(true, predicted):
    trueSize = float(len(true))
    predSize = float(len(predicted))
    p1 = 0.
    p2 = 0.
    for cStar in true:
        cStar = np.array(cStar)
        tempArr = []
        for c in predicted:
            cS = cStar
            cC = c
            if(len(c) > len(cStar)):
                cS = np.array([-1] * len(c))
                cS[:len(cStar)] = cStar
#                 print("RESHAPING")
            if(len(cStar) > len(c)):
                cC = np.array([-1] * len(cStar))
                cC[:len(c)] = c
#                 print("RESHAPING")
#             print("SHAPES: ", cS.shape, cC.shape)
            tempArr.append(delta(cS, cC))
        p1 += np.max(tempArr)
    for cPred in predicted:
        cPred = np.array(cPred)
        tempArr = []
        for cStar in true:
            cS = cStar
            cC = cPred
            if(len(cPred) > len(cStar)):
                cS = np.array([-1] * len(cPred))
                cS[:len(cStar)] = cStar
#                 print("RESHAPING")
            if(len(cStar) > len(cPred)):
                cC = np.array([-1] * len(cStar))
                cC[:len(cPred)] = cPred
#                 print("RESHAPING")
#             print("SHAPES: ", cS.shape, cC.shape)
            tempArr.append(delta(cS, cC))
        p2 += np.max(tempArr)

#     p2 += np.max([delta(cStar, cC) for cStar in true])
    score = (1/(2*trueSize))*p1 + (1/(2*predSize))*p2
    return score

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
