import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx
import itertools
from tqdm import tqdm
import random

from utils import *

def find_path_nearest_neighbor(G, start_node):
    path = [start_node]
    current = start_node
    total_cost = 0

    while len(path) < len(G.nodes):
        neighbors = [(n, G[current][n]['weight']) for n in G.neighbors(current) if n not in path]
        if not neighbors:
            break  # No more neighbors to visit
        next_node, cost = min(neighbors, key=lambda x: x[1])
        path.append(next_node)
        total_cost += cost
        current = next_node

    return path, total_cost


def main():
    df = load_process_data()

    X = df.iloc[:, 2:].values

    G = create_graph(df)

    best_path = None
    best_cost = float('inf')

    for start_node in tqdm(G.nodes(), desc="Finding best path"):
        path, cost = find_path_nearest_neighbor(G, start_node)
        if cost < best_cost:
            tqdm.write(f"Found better path with cost {cost}")
            best_cost = cost
            best_path = path

    print(f"Best path: {best_path}")
    print(f"Total cost: {best_cost}")

    # Save the best path by using df and 'song', 'bpm' and 'key' columns
    # So it means reordering using the order in best_path
    df = df.set_index('song')
    df = df.loc[best_path]
    df = df.reset_index()
    # only keep the columns we need
   

    # Add a column with the distance from the previous song (ignore first song)
    d = []
    for i in range(1, len(df)):
        distance = compute_distance(df.loc[i-1], df.loc[i])
        d.append(distance)
    df['distance'] = [0] + d

    df = df[['song', 'bpm', 'key', 'distance']]
    df.to_csv('best_path.csv', index=False)

if __name__ == "__main__":
    main()