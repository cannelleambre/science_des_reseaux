import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from haversine import haversine, Unit
import math
from colorateur import color_csv

def run_dbscan(csv_file, max_pir_threshold):
    # Lecture du CSV
    dtype_dict = {
        'LAT': np.float32,
        'LON': np.float32
    }
    donnees = pd.read_csv(csv_file, delimiter=",", dtype=dtype_dict)

    # Calcul des clusters avec DBSCAN
    coords = np.radians(donnees[['LAT', 'LON']].values.astype(np.float32))
   
     # Calcul des clusters avec DBSCAN
    coords = np.radians(donnees[['LAT', 'LON']].values.astype(np.float32))
    dbscan = DBSCAN(eps=45/6371.0, min_samples=10)  # eps est la distance maximale entre les points dans un cluster
    labels = dbscan.fit_predict(coords)

   # Assignation des clusters
    n = 1
    cluster_map = np.zeros(len(donnees), dtype=np.int32)
    sorted_indices = donnees['PIR'].sort_values(ascending=False).index

    for index in sorted_indices:
        if cluster_map[index] == 0:
            cluster_map[index] = n
            voisins = []
            for i, label in enumerate(labels):
                if label == labels[index] and i != index:
                    voisins.append(i)
            pir_somme = donnees.at[index, 'PIR']
            for voisin in voisins:
                pir_somme += donnees.at[voisin, 'PIR']
                if pir_somme > max_pir_threshold:
                    break
            if pir_somme <= max_pir_threshold:
                cluster_map[voisins] = n            
            n += 1

    donnees['cluster'] = cluster_map
    # Export des résultats
    donnees.to_csv("res/res_clusters_dbscan.csv", sep=',', index=False)

    # Génération des couleurs
    color_csv("res/res_clusters_dbscan.csv")