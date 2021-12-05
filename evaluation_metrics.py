import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

def cluster_embeddings(data):
    k_means = KMeans(n_clusters= 3)
    pca = PCA(2)
    pca_df = pca.fit_transform(data)
    
    label = k_means.fit_predict(data)
    u_labels = np.unique(label)

    for u in u_labels:
        plt.scatter(pca_df[label==u , 0] , pca_df[label==u , 1] , label = u)
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



if __name__ == "__main__":
    with open('100_artist_tfidf_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
        embeddings = []
        for song_list in data:
            embeddings.append(song_list[5])
        cluster_embeddings(embeddings)
        print (average_cosine_distance(embeddings))

    



