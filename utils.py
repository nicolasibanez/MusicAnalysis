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
    # Expanded circle of fifths to include relative minor keys
    circle_of_fifths = [
        'C', 'Am', 'G', 'Em', 'D', 'Bm', 'A', 'F#m', 
        'E', 'C#m', 'B', 'G#m', 'F#', 'D#m', 'Db', 'Bbm', 
        'Ab', 'Fm', 'Eb', 'Cm', 'Bb', 'Gm', 'F', 'Dm'
    ]

    # Check if keys are the same
    if key1 == key2:
        return 0

    # Check for relative or parallel keys
    if key1 in circle_of_fifths and key2 in circle_of_fifths:
        pos1 = circle_of_fifths.index(key1)
        pos2 = circle_of_fifths.index(key2)
        distance = min(abs(pos1 - pos2), 24 - abs(pos1 - pos2))  # Account for the circular nature

        # Relative or parallel keys (e.g., C Major and A Minor or C Major and C Minor)
        if distance == 3 or (key1[:-1] == key2[:-1] and key1[-1] != key2[-1]):
            return 1

        # Perfect fifth apart
        elif distance == 7 or distance == 17:
            return 1

        # Other cases
        else:
            return 3
    else:
        return 3  # Default case if key not found

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
            if i != j and key_d <= 1:
                distance = compute_distance(row_i, row_j)
                G.add_edge(row_i['song'], row_j['song'], weight=distance)
    return G