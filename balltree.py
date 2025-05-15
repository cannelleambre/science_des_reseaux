import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
import numpy as np
from haversine import haversine, Unit
import math
from colorateur import color_csv
from calculs_clusters_stats import calcul_nb_clusters, nb_users_par_cluster

def run_ball_tree(csv_file, threshold_pir):
    # Lecture du CSV
    dtype_dict = {
        'LAT': np.float32,
        'LON': np.float32
    }
    donnees = pd.read_csv("generated.csv", delimiter=",", dtype=dtype_dict)

    # Initialisation des colonnes
    donnees['nb_voisins'] = np.zeros(len(donnees), dtype=np.int32)
    donnees['cluster'] = np.zeros(len(donnees), dtype=np.int32)
    donnees['VID'] = [[] for _ in range(len(donnees))]

    # Calcul des voisins avec BallTree
    coords = np.radians(donnees[['LAT', 'LON']].values.astype(np.float32))
    tree = BallTree(coords, metric='haversine', leaf_size=40)
    radius = 45 / 6371.0

    indices = tree.query_radius(coords, r=radius)
    donnees['nb_voisins'] = np.array([len(neighs) - 1 for neighs in indices], dtype=np.int32)
    donnees['VID'] = [list(neighs[neighs != i]) for i, neighs in enumerate(indices)] # voisins id
    print(donnees)
    # Assignation des clusters
    n = 1
    cluster_map = np.zeros(len(donnees), dtype=np.int32)
    sorted_indices = donnees['nb_voisins'].sort_values(ascending=False).index

    for index in sorted_indices:
        if cluster_map[index] == 0:
            cluster_map[index] = n
            voisins = donnees.at[index, 'VID']
            pir_somme = donnees.at[index, 'PIR']
            for voisin in voisins:
                pir_somme += donnees.at[voisin, 'PIR']
                if pir_somme > threshold_pir:
                    break
            if pir_somme <= threshold_pir:
                cluster_map[voisins] = n            
            n += 1

    donnees['cluster'] = cluster_map

    # Export des résultats
    donnees.to_csv("res/res_clusters_ball_tree.csv", sep=',', index=False)
    #generation des couleurs
    color_csv("res/res_clusters_ball_tree.csv")