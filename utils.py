import pandas as pd
import numpy as np
import networkx as nx


def load_process_data(path = 'features.csv'):
    df = pd.read_csv(path)

    # STD scale every features apart from song and bpm : 
    features = df.drop(['song', 'bpm', 'key'], axis=1)
    features = (features - features.mean()) / features.std()
    features = features.fillna(0)
    df[features.columns] = features

    return df


def key_distance(key1, key2):
    # Major and minor keys in logical order
    major_keys = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    minor_keys = ['Am', 'A#m', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m']

    # Check if keys are the same
    if key1 == key2:
        return 0

    # Determine if the keys are major or minor
    is_minor1 = 'm' in key1
    is_minor2 = 'm' in key2

    # Relative Minor/Major check
    if is_minor1 != is_minor2:
        minor_key = key1 if is_minor1 else key2
        major_key = key1[:-1] if is_minor1 else key1
        major_key_other = key2[:-1] if is_minor2 else key2

        # Convert minor to its relative major and compare
        relative_major = major_keys[(minor_keys.index(minor_key) + 3) % 12]
        if major_key == relative_major or major_key_other == relative_major:
            return 0.5
        elif major_key == major_key_other: # Major to its parallel minor or vice versa
            return 3
        else:
            return 7

    # Both keys are of the same type (either both major or both minor)
    else:
        index1 = major_keys.index(key1) if not is_minor1 else minor_keys.index(key1)
        index2 = major_keys.index(key2) if not is_minor2 else minor_keys.index(key2)
        diff = abs(index1 - index2)

        if diff in [5, 7]:
            return 2
        else:
            return 7

# Distance between two rows of df :
def compute_distance(row_i, row_j):
    # L1 distance for bpm : 
    distance = np.abs(row_i['bpm'] - row_j['bpm']).astype(np.float64)

    # key_distance for key :
    key_d = key_distance(row_i['key'], row_j['key'])
    
    distance += key_d

    # L2 norm for the rest of the features :
    columns = row_i.index
    for column in columns:
        if column not in ['song', 'bpm', 'key']:
            l2_dist = np.square(row_i[column] - row_j[column])
            distance += l2_dist
    
    return distance

def create_graph(df):
    G = nx.Graph()
    
    # Add nodes with the BPM attribute
    for index, row in df.iterrows():
        G.add_node(row['song'], bpm=row['bpm'])
    
    # Add edges with a weight attribute if the BPM difference is within the threshold
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i != j: #and abs(row_i['bpm'] - row_j['bpm']) <= threshold:
                distance = compute_distance(row_i, row_j)
                G.add_edge(row_i['song'], row_j['song'], weight=distance)
                
    return G

def create_graph_key_constraint(df):
    G = nx.Graph()
    
    # Add nodes with the BPM attribute
    for index, row in df.iterrows():
        G.add_node(row['song'], bpm=row['bpm'])
    
    # Add edges with a weight attribute if the BPM difference is within the threshold
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            key_d = key_distance(row_i['key'], row_j['key'])
            if i != j and key_d <= 2:
                distance = compute_distance(row_i, row_j)
                G.add_edge(row_i['song'], row_j['song'], weight=distance)
    return G