import pickle
from datetime import datetime

import pandas as pd
import numpy as np

from wellcomeml.ml import TextClustering

from nutrition_labels.grant_data_processing import clean_grants_data

def get_cluster_keys(cluster):
    """
    Get information about each cluster:
    - keywords
    - size cluster
    - centroid of cluster
    """

    cluster_kws = {}
    cluster_numbers = {}
    cluster_centroids = {}
    for cluster_num in list(set(cluster.cluster_ids)):
        cluster_idx = [i for i, c_num in enumerate(cluster.cluster_ids) if c_num==cluster_num]
        cluster_kws[cluster_num] = cluster.cluster_kws[cluster_idx[0]]
        cluster_numbers[cluster_num] = sum(cluster.cluster_ids==cluster_num)
        cluster_centroids[cluster_num] = np.mean(cluster.reduced_points[cluster_idx], axis = 0)
        
    return cluster_kws, cluster_centroids, cluster_numbers

def save_cluster(cluster, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(cluster, file=f)

def load_cluster(cluster_file):
    with open(cluster_file, 'rb') as f:
        cluster = pickle.load(f)
    return cluster

if __name__ == '__main__':

    datestamp = datetime.now().date().strftime('%y%m%d')
    
    tech_grants = pd.read_csv("data/processed/ensemble_results.csv")
    grant_data = pd.read_csv("data/raw/wellcome-grants-awarded-2005-2019.csv")

    output_file = f'models/clustering/grants_clusters_{datestamp}.pkl'
    tech_output_file = f'models/clustering/tech_grants_clusters_{datestamp}.pkl'
    grant_data_output_file = f'data/processed/cluster_grant_data_{datestamp}.csv'

    # Process grants data for clustering - clean, merge with tech grants, and
    # remove duplicate 6 digits, but make sure to keep the ones that are identified as tech grants
    grant_data = clean_grants_data(grant_data)
    grants_text = grant_data[['Title', 'Grant Programme:Title', 'Description']].agg(
                '. '.join, axis=1
                ).tolist()
    grant_data['Grant Text'] = grants_text
    tech_grant_ids = tech_grants['Internal ID'].tolist()
    grant_data['Tech grant?'] = grant_data['Internal ID'].isin(tech_grant_ids)
    grant_data.sort_values(by=['Tech grant?'], ascending=False, inplace=True)
    grant_data.drop_duplicates(subset=['Internal ID 6 digit'], inplace=True)

    # Cluster on all grant data
    X = grant_data['Grant Text'].tolist()
    params = {
        'clustering': {'min_samples': 20, 'eps': 0.12},
        'reducer': {'metric': 'cosine', 'min_dist': 0.15, 'n_neighbors': 30}}
    cluster = TextClustering(
        embedding='tf-idf', reducer='umap', clustering='dbscan', params=params
        )
    cluster.fit(X)
    save_cluster(cluster, output_file)

    # Cluster on the tech grants only
    tech_grant_data = grant_data.loc[grant_data['Tech grant?']]
    X_tech = tech_grant_data['Grant Text'].tolist()
    params = {
        'clustering': {'min_samples': 8, 'eps': 0.15},
        'reducer': {'metric': 'cosine', 'min_dist': 0, 'n_neighbors': 30}}
    tech_cluster = TextClustering(
        embedding='tf-idf', reducer='umap', clustering='dbscan', params=params)
    tech_cluster.fit(X_tech)
    save_cluster(tech_cluster, tech_output_file)

    # Add cluster information to grant data, and save
    tech_grant_ids = tech_grant_data['Internal ID'].tolist()
    cluster_reduced_points = []
    tech_cluster_reduced_points = []
    tech_clusters = []
    t_i = 0
    for i, row in grant_data.reset_index().iterrows():
        cluster_reduced_points.append(cluster.reduced_points[i])
        if row['Internal ID'] in tech_grant_ids:
            tech_clusters.append(tech_cluster.cluster_ids[t_i])
            tech_cluster_reduced_points.append(list(tech_cluster.reduced_points[t_i]))
            t_i += 1
        else:
            tech_clusters.append(None)
            tech_cluster_reduced_points.append(None)
    grant_data['Cluster number'] = cluster.cluster_ids
    grant_data['Tech cluster number'] = tech_clusters
    grant_data['Cluster reduction x'] = [c[0] for c in cluster_reduced_points]
    grant_data['Cluster reduction y'] = [c[1] for c in cluster_reduced_points]
    grant_data['Tech cluster reduction x'] = [c[0] if c else None for c in tech_cluster_reduced_points]
    grant_data['Tech cluster reduction y'] = [c[1] if c else None for c in tech_cluster_reduced_points]

    grant_data.to_csv(grant_data_output_file)


