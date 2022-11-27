import argparse
import os
import sklearn.cluster
import sklearn.mixture
import sklearn.decomposition
import numpy as np
import json
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import tqdm

first_pc = 1
last_pc = 50

if __name__ == '__main__':

    ## parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--embed_dir', type=str)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--cluster_method', type=str, default='kmeans')
    parser.add_argument('--subset_size', type=int, default=None)

    args = parser.parse_args()

    embed_dir = args.embed_dir
    k = args.k
    cluster_method = args.cluster_method
    if embed_dir[-1] == '/':
        embed_dir = embed_dir[:-1]
    subset_size = args.subset_size
    
    # loading the embeddings and texts
    f_prefixes = sorted([f.split('.')[0] for f in os.listdir(embed_dir) if f.endswith('.npy')], key=lambda x: int(x))
    all_embeddings, all_texts = [], []
    for f in tqdm.tqdm(f_prefixes):
        arr = np.load(os.path.join(embed_dir, f + '.npy'))
        all_embeddings.extend(np.load(os.path.join(embed_dir, f + '.npy')))
        all_texts.extend(json.load(open(os.path.join(embed_dir, f + '.json'))))
        if len(all_embeddings) >= subset_size:
            break
    all_embeddings = np.array(all_embeddings)[:subset_size]
    all_texts = all_texts[:subset_size]
    print(f'finished loading {len(all_embeddings)} embeddings')

    # first run PCA
    pca = sklearn.decomposition.PCA(n_components=1+last_pc)

    # fit the PCA model to the embeddings
    all_embs = pca.fit_transform(all_embeddings)[:, list(range(first_pc,last_pc+1))]
    print('finished PCA')

    # GMM clustering
    # defining the clustering model
    if cluster_method == 'gmm':
        cluster = sklearn.mixture.GaussianMixture(n_components=k, covariance_type='full')
    elif cluster_method == 'kmeans':
        cluster = KMeans(n_clusters=k)
    
    cluster.fit(all_embs)
    if cluster_method == 'gmm':
        centers = cluster.means_
    elif cluster_method == 'kmeans':
        centers = cluster.cluster_centers_

    print('finished clustering')
    data_center_distance = euclidean_distances(all_embs, centers)
    cluster_idxes = cluster.predict(all_embs)

    print('finished predicting probabilities')
    center_pairwise_distances = euclidean_distances(centers, centers)


    # saving the results
    save_dir = 'clusters/' + os.path.basename(embed_dir) + f'_k={k}' + f'_method={cluster_method}' + f'_subset={subset_size}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json.dump(all_texts, open(os.path.join(save_dir, 'all_texts.json'), 'w'))
    np.save(os.path.join(save_dir, 'data_center_distance'), data_center_distance)
    np.save(os.path.join(save_dir, 'centers'), centers)
    np.save(os.path.join(save_dir, 'center_pairwise_distance'), center_pairwise_distances)
    np.save(os.path.join(save_dir, 'cluster_idxes'), cluster_idxes)
