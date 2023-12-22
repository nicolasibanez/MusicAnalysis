import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import *


def main():
    path = 'features.csv'
    df = load_process_data(path=path)
    # Take all columns apart from song, key, bpm : 
    X = df.drop(['song', 'key', 'bpm'], axis=1).values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot the first two principal components
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='black', s=20)
    plt.title('PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # save : 
    plt.savefig('pca.png')



if __name__ == "__main__":
    main()