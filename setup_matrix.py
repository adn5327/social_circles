import numpy as np

NUM_NODES = 4039

def read_input(filename):
    arr = np.zeros((NUM_NODES, NUM_NODES), dtype='int')
    with open(filename, 'r') as f:
        for line in f:
            x = line.rstrip().split(' ')
            src = int(x[0])
            dest = int(x[1])
            arr[src][dest] = 1
            arr[dest][src] = 1
    return arr

if __name__ == "__main__":
    read_input('data/facebook_combined.txt')
