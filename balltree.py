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




def run_simulation_ball_tree(csv_file):
    #1 Gbps
    run_ball_tree(csv_file, 1000)
    nb_cluster_mean_1gbps = calcul_nb_clusters("res/res_clusters_ball_tree.csv")
    nb_usrs_mean_1gbps = nb_users_par_cluster("res/res_clusters_ball_tree.csv")
    #2 Gbps
    run_ball_tree(csv_file, 2000)
    nb_cluster_mean_2gbps = calcul_nb_clusters("res/res_clusters_ball_tree.csv")
    nb_usrs_mean_2gbps = nb_users_par_cluster("res/res_clusters_ball_tree.csv")
    #4 Gbps
    run_ball_tree(csv_file, 4000)
    nb_cluster_mean_4gbps = calcul_nb_clusters("res/res_clusters_ball_tree.csv")
    nb_usrs_mean_4gbps = nb_users_par_cluster("res/res_clusters_ball_tree.csv")

    # Créer un DataFrame avec les informations
    data = {
        'Débit': ['1 Gbps', '2 Gbps', '4 Gbps'],
        'Nb de clusters': [nb_cluster_mean_1gbps, nb_cluster_mean_2gbps, nb_cluster_mean_4gbps],
        'Nb d\'utilisateurs par cluster': [nb_usrs_mean_1gbps, nb_usrs_mean_2gbps, nb_usrs_mean_4gbps]
    }

    df = pd.DataFrame(data)

    # Enregistrer le DataFrame dans un fichier CSV
    df.to_csv('stats_ball_tree.csv', index=False)

    print("Pour 1 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_1gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_1gbps))

    print("Pour 2 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_2gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_2gbps))

    print("Pour 4 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_4gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_4gbps))


run_simulation_ball_tree("test_small_csav_500.csv")