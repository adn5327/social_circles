import sys

import girvan

def runner(graph_fname, exact=False):
    x = girvan.CommunityDetectionGraph(graph_fname, exact)
    x.girvan_newman()

if __name__ == "__main__":
    runner(sys.argv[1], False)

