import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cluster_embeddings(data):
    X = pd.read_csv('HealthClusteringData.csv')
    k_means = KMeans(n_clusters= 10)
    pca = PCA(2)
    pca_df = pca.fit_transform(X)
    
    labels = k_means.fit_predict(X)
    u_labels = np.unique(labels)

    for u in u_labels:
        plt.scatter(pca_df[labels==u , 0] , pca_df[labels==u , 1] , labels = u)
    plt.legend()
    plt.show()


def average_cosine_distance(data):
    total_pairs = 0
    total_distance = 0
    for i in range(len(data)-1):
        for j in range(i+1, len(data)):
            total_pairs += 1
            embedding_1 = data[i]
            embedding_2 = data[j]
            distance = np.dot(embedding_1, embedding_2)/(np.linalg.norm(embedding_1)*np.linalg.norm(embedding_2))
            total_distance += distance

    return total_distance/total_pairs




