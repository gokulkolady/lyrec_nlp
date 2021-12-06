import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import torch

def cluster_embeddings(data, pca_df):
    k_means = KMeans(n_clusters= 3)
    
    label = k_means.fit_predict(data)
    u_labels = np.unique(label)

    for u in u_labels:
        plt.scatter(pca_df[label==u , 0] , pca_df[label==u , 1] , label = u)
    plt.legend()
    plt.show()

    

def get_reduced_embedding(data, plot_embedding = False):
    label = []
    embeddings = []
    freq = {}
    for info in data:
        artist = info[1]
        if artist not in freq:
            freq[artist] = 1
        else:
            freq[artist] += 1
        label.append(artist)
        embeddings.append(info[5])

    pca = PCA(2)
    pca_df = pca.fit_transform(embeddings)
    if plot_embedding:
        label = np.array(label)
        u_labels = []
        for _ in range(8):
            max_artist = max(freq, key = freq.get)
            freq.pop(max_artist)
            u_labels.append(max_artist)
            if freq == {}:
                break
        for u in u_labels:
            ix = np.where(label==u)
            plt.scatter(pca_df[ix, 0] , pca_df[ix, 1] , label = u)
        plt.legend()
        plt.show()
    return pca_df


def average_cosine_distance(data):
    total_pairs = 0
    total_distance = 0
    for i in range(len(data)-1):
        for j in range(i+1, len(data)):
            total_pairs += 1
            embedding_1 = data[i]
            embedding_2 = data[j]
            dot_product = np.dot(embedding_1, embedding_2)
            norm_1 = np.linalg.norm(embedding_1)
            norm_2 = np.linalg.norm(embedding_2)
            if norm_1 == 0:
                norm_1 = 0.0000000001
            if norm_2 == 0:
                norm_2 = 0.0000000001
            distance = dot_product/(norm_1*norm_2)
            total_distance += distance

    return total_distance/total_pairs



if __name__ == "__main__":
    # with open('eval_data/Happy_Hits_evaluation_extended.pkl', 'rb') as f:
    #     with open('../eval_data/Sad_Hour_evaluation_extended.pkl', 'rb') as f2:
    with open('unique_song_dataset_extended.pkl', 'rb') as f:
            # data = pickle.load(f)
            data = pickle.load(f)
            embeddings = []
            all_scores = []
            for song_list in data:
                valence = song_list[2].detach().tolist()[0] - 50
                # valence = song_list[3]*100 - 50
                embeddings.append(song_list[1]+ [valence])

            # pca_df = get_reduced_embedding(data, True)
            # cluster_embeddings(embeddings, pca_df)
            print (average_cosine_distance(embeddings))

    



