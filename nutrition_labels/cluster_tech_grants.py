import pickle
from datetime import datetime
import configparser
from argparse import ArgumentParser
import os 

import pandas as pd
import numpy as np

from wellcomeml.ml import TextClustering

from nutrition_labels.utils import clean_grants_data

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
        cluster_centroids[cluster_num] = list(np.mean(cluster.reduced_points[cluster_idx], axis = 0))
        
    return cluster_kws, cluster_centroids, cluster_numbers

def save_cluster(cluster, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(cluster, file=f)

def load_cluster(cluster_file):
    with open(cluster_file, 'rb') as f:
        cluster = pickle.load(f)
    return cluster

def clean_for_clustering(grant_data, tech_grants, tech_grant_id='Internal ID'):
    # Process grants data for clustering - clean, merge with tech grants, and
    # remove duplicate 6 digits, but make sure to keep the ones that are identified as tech grants
    grant_data = clean_grants_data(grant_data)
    grants_text = grant_data[['Description']].agg(
                '. '.join, axis=1
                ).tolist()
    grant_data['Grant Text'] = grants_text
    
    # Short amounts of text cause bad clusters (end up clustering on grant programme names)
    grant_data = grant_data[grant_data['Grant Text'].map(len)>100]
    
    # These grants have short descriptions which are all quite different, but 
    # end up clustering together because of these words
    grant_data = grant_data[~grant_data['Grant Text'].str.contains('Student Elective Prize for')]
    grant_data = grant_data[~grant_data['Title'].str.contains('Vacation Scholarships')]
    # For this the description isn't about the project but the scholarship
    grant_data = grant_data[~grant_data['Title'].str.contains('Biomedical Vacation Scholarship')]
                                      
    tech_grant_ids = tech_grants[tech_grant_id].tolist()
    grant_data['Tech grant?'] = grant_data['Internal ID'].isin(tech_grant_ids)
    grant_data.sort_values(by=['Tech grant?'], ascending=False, inplace=True)
    grant_data.drop_duplicates(subset=['Internal ID 6 digit'], inplace=True)
    
    return grant_data

def old_clean_for_clustering(grant_data, tech_grants):
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

    return grant_data

if __name__ == '__main__':

    datestamp = datetime.now().date().strftime('%y%m%d')

    parser = ArgumentParser()
    parser.add_argument(
        '--config_path',
        help='Path to config file',
        default='configs/clustering/2020.11.25.ini'
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    tech_grants_file_name = config["data"]["tech_grants_file_name"]
    tech_grants = pd.read_csv(tech_grants_file_name)
    grant_data = pd.read_csv("data/raw/wellcome-grants-awarded-2005-2019.csv")

    output_folder = f'models/clustering/{datestamp}/'
    grant_data_output_file = f'data/processed/clustering/cluster_grant_data_{datestamp}.csv'

    if args.config_path == 'configs/clustering/2020.09.16.ini':
        # For reproducibility of the older results
        grant_data = old_clean_for_clustering(grant_data, tech_grants)
    else:
        try:
            tech_grant_id = config["data"]["tech_grant_id_col"]
            tech_grants = tech_grants[tech_grants['Tech grant prediction'] == 1]
            grant_data = clean_for_clustering(grant_data, tech_grants, tech_grant_id=tech_grant_id)
        except: 
            grant_data = clean_for_clustering(grant_data, tech_grants)
    
    # Cluster on all grant data
    X = grant_data['Grant Text'].tolist()
    params = {
        'clustering': {
            'min_samples': int(config["cluster_all"]["cluster_min_samples"]),
            'eps': float(config["cluster_all"]["cluster_eps"])},
        'reducer': {
            'metric': config["cluster_all"]["reducer_metric"],
            'min_dist': float(config["cluster_all"]["reducer_min_dist"]),
            'n_neighbors': int(config["cluster_all"]["reducer_n_neighbors"])
            }}
    cluster = TextClustering(
        embedding=config["cluster_general"]["embedding"],
        reducer=config["cluster_general"]["reducer"],
        clustering=config["cluster_general"]["clustering"], params=params
        )
    cluster.fit(X)
    cluster.save(os.path.join(output_folder, 'all'), components=['reduced_points', 'clustering_class'])

    # Cluster on the tech grants only
    tech_grant_data = grant_data.loc[grant_data['Tech grant?']]
    X_tech = tech_grant_data['Grant Text'].tolist()
    params = {
        'clustering': {
            'min_samples': int(config["cluster_tech"]["cluster_min_samples"]),
            'eps': float(config["cluster_tech"]["cluster_eps"])},
        'reducer': {
            'metric': config["cluster_tech"]["reducer_metric"],
            'min_dist': float(config["cluster_tech"]["reducer_min_dist"]),
            'n_neighbors': int(config["cluster_tech"]["reducer_n_neighbors"])
            }}
    tech_cluster = TextClustering(
        embedding=config["cluster_general"]["embedding"],
        reducer=config["cluster_general"]["reducer"],
        clustering=config["cluster_general"]["clustering"], params=params
        )
    tech_cluster.fit(X_tech)
    tech_cluster.save(os.path.join(output_folder, 'tech'), components=['reduced_points', 'clustering_class'])

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

    grant_data = grant_data.copy()[[
        'Internal ID', 'Tech grant?', 'Cluster number', 'Tech cluster number',
        'Cluster reduction x', 'Cluster reduction y',
        'Tech cluster reduction x', 'Tech cluster reduction y'
        ]]

    grant_data.to_csv(grant_data_output_file)

    cluster_kws, cluster_centroids, cluster_numbers = get_cluster_keys(cluster)
    with open(f'data/processed/clustering/cluster_info_{datestamp}.txt', 'w') as file:
        file.write(str(cluster_kws))
        file.write('\n')
        file.write(str(cluster_centroids))
        file.write('\n')
        file.write(str(cluster_numbers))
        file.write('\n')

    tech_cluster_kws, tech_cluster_centroids, tech_cluster_numbers = get_cluster_keys(tech_cluster)
    with open(f'data/processed/clustering/tech_cluster_info_{datestamp}.txt', 'w') as file:
        file.write(str(tech_cluster_kws))
        file.write('\n')
        file.write(str(tech_cluster_centroids))
        file.write('\n')
        file.write(str(tech_cluster_numbers))
        file.write('\n')


