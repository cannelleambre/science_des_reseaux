import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from haversine import haversine, Unit
import math


def run_ball_tree(csv_file):
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
            cluster_map[voisins] = n
            n += 1

    donnees['cluster'] = cluster_map

    # Assignation des clusters
    n = 1
    cluster_map = np.zeros(len(donnees), dtype=np.int32)
    sorted_indices = donnees['nb_voisins'].sort_values(ascending=False).index

    for index in sorted_indices:
        if cluster_map[index] == 0:
            cluster_map[index] = n
            voisins = donnees.at[index, 'VID']
            cluster_map[voisins] = n
            n += 1

    donnees['cluster'] = cluster_map

# Génération des couleurs
    def generate_color(i, total):
        # Utiliser l'index d'origine pour déterminer la couleur
        hue = (i / total) % 1.0
        saturation = 0.8
        value = 0.9
        rgb = np.array([hue, saturation, value])
        # Convertir HSV en RGB
        h, s, v = rgb
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            rgb = [c, x, 0]
        elif h < 2/6:
            rgb = [x, c, 0]
        elif h < 3/6:
            rgb = [0, c, x]
        elif h < 4/6:
            rgb = [0, x, c]
        elif h < 5/6:
            rgb = [x, 0, c]
        else:
            rgb = [c, 0, x]
        
        rgb = [(int((r + m) * 255)) for r in rgb]
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    # Ajout de la colonne couleur
    donnees['color'] = 'white'  # Couleur par défaut pour les non-clusterisés
    clusters_uniques = sorted(donnees['cluster'].unique())
    clusters_uniques = [c for c in clusters_uniques if c != 0]  # Exclure le cluster 0
    nombre_clusters = len(clusters_uniques)

    for i, cluster in enumerate(clusters_uniques):
        donnees.loc[donnees['cluster'] == cluster, 'color'] = generate_color(i, nombre_clusters)


    # Export des résultats
    donnees.to_csv("res_clusters.csv", sep=',', index=False)

def calcul_nb_clusters(csv_file):

    # Charger le fichier CSV
    df = pd.read_csv(csv_file)

    # Compter le nombre de clusters uniques
    num_clusters = df['cluster'].nunique()

    print(num_clusters )
    #chercher nb cluster + nb_moyen/clusters
    # etat de l'art à faire

def nb_users_par_cluster(fichier):
    # Charger le fichier CSV
    df = pd.read_csv(fichier)

    # Compter le nombre d'utilisateurs par cluster
    nb_users_par_cluster = df['cluster'].value_counts()

    # Calculer la moyenne du nombre d'utilisateurs par cluster
    moyenne_nb_users = nb_users_par_cluster.mean()

    print( moyenne_nb_users)


run_ball_tree("generated.csv")
calcul_nb_clusters("res_clusters.csv")
nb_users_par_cluster("res_clusters.csv")