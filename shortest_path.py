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

    while len(path) < min(len(G.nodes), 25):
        neighbors = [(n, G[current][n]['weight']) for n in G.neighbors(current) if n not in path]
        if not neighbors:
            break  # No more neighbors to visit
        next_node, cost = min(neighbors, key=lambda x: x[1])
        path.append(next_node)
        total_cost += cost
        current = next_node

    return path, total_cost

def find_best_path(df, nb_nodes = 25):
    # G = create_graph(df)
    G = create_graph_key_constraint(df)

    best_path = None
    best_cost = float('inf')

    for start_node in tqdm(G.nodes(), desc="Finding best path"):
        path, cost = find_path_nearest_neighbor(G, start_node)
        if len(path) >= nb_nodes and cost < best_cost:
            tqdm.write(f"Found better path with cost {cost}")
            best_cost = cost
            best_path = path

    print(f"Best path: {best_path}")
    print(f"Total cost: {best_cost}")
    return best_path

def main():
    df = load_process_data()

    k = 0
    best_path = find_best_path(df, 25)
    while len(best_path) >= 25:
        # Other initializations...

        df_k = df.copy()

        # Save the best path by using df and 'song', 'bpm' and 'key' columns
        # So it means reordering using the order in best_path
        df_k = df_k.set_index('song')
        df_k = df_k.loc[best_path]
        df_k = df_k.reset_index()
        # only keep the columns we need
        # Add a column with the distance from the previous song (ignore first song)
        d = []
        for i in range(1, len(df_k)):
            distance = compute_distance(df_k.loc[i-1], df_k.loc[i])
            d.append(distance)
        df_k['distance'] = [0] + d

        df_k = df_k[['song', 'bpm', 'key', 'distance']]
        df_k.to_csv('best_path_' + str(k) + '.csv', index=False)

        # Remove the songs in best_path from df
        df = df[~df['song'].isin(best_path)]

        best_path = find_best_path(df, 25)
        k += 1
        print('------------------------------------\n')



if __name__ == "__main__":
    main()