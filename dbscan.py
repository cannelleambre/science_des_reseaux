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
    donnees_temp = donnees
    # Initialisation des colonnes
    donnees['nb_voisins'] = np.zeros(len(donnees), dtype=np.int32)
    donnees['cluster'] = np.zeros(len(donnees), dtype=np.int32)
    donnees['VID'] = [[] for _ in range(len(donnees))]


    # Calcul des clusters avec DBSCAN
    coords = np.radians(donnees[['LAT', 'LON']].values.astype(np.float32))
   
     # Calcul des clusters avec DBSCAN
    coords = np.radians(donnees[['LAT', 'LON']].values.astype(np.float32))
    dbscan = DBSCAN(eps=45, min_samples=1)  # eps est la distance maximale entre les points dans un cluster
    labels = dbscan.fit_predict(coords)


    donnees['cluster'] = dbscan.labels_
    donnees['nb_voisins'] = donnees.groupby('cluster')['cluster'].transform('count') - 1
    donnees = donnees.reset_index()
    donnees['VID'] = donnees.apply(lambda row: donnees[(donnees['cluster'] == row['cluster']) & (donnees['index'] != row['index'])]['index'].tolist(), axis=1)
    if donnees_temp.equals(donnees):
        print("J'EN AI MAAAAAAARES")
    # Assignation des clusters
    n = 1
    cluster_map = np.zeros(len(donnees), dtype=np.int32)
    sorted_indices = donnees['PIR'].sort_values(ascending=False).index

    for index in sorted_indices:
        if cluster_map[index] == 0:
            cluster_map[index] = n
            voisins = donnees.at[index, 'VID']
            pir_somme = donnees.at[index, 'PIR']
            for voisin in voisins:
                pir_somme += donnees.at[voisin, 'PIR']
                if pir_somme > max_pir_threshold:
                    break
            if pir_somme <= max_pir_threshold:
                for voisin in voisins:
                    cluster_map[voisin] = n
            n += 1

    donnees['cluster'] = cluster_map
    # Export des résultats
    donnees.to_csv("res/res_clusters_dbscan.csv", sep=',', index=False)

    # Génération des couleurs
    color_csv("res/res_clusters_dbscan.csv")