
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from haversine import haversine, Unit
import math


def calcul_nb_clusters(csv_file):

    # Charger le fichier CSV
    df = pd.read_csv(csv_file)

    # Compter le nombre de clusters uniques
    num_clusters = df['cluster'].nunique()

    return(num_clusters )
    #chercher nb cluster + nb_moyen/clusters
    # etat de l'art à faire

def nb_users_par_cluster(fichier):
    # Charger le fichier CSV
    df = pd.read_csv(fichier)

    # Compter le nombre d'utilisateurs par cluster
    nb_users_par_cluster = df['cluster'].value_counts()

    # Calculer la moyenne du nombre d'utilisateurs par cluster
    moyenne_nb_users = nb_users_par_cluster.mean()

    return( moyenne_nb_users)
def run_dbscan(csv_file, max_pir_threshold):
    # Lecture du CSV
    dtype_dict = {
        'LAT': np.float32,
        'LON': np.float32
    }
    donnees = pd.read_csv(csv_file, delimiter=",", dtype=dtype_dict)

    # Calcul des clusters avec DBSCAN
    coords = np.radians(donnees[['LAT', 'LON']].values.astype(np.float32))
    #appel à dbscan
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(coords)
    clusters = clustering.labels_

    indices = []
    for i, label in enumerate(clusters):
        if label == -1:
            continue
        indices.append(np.where(clusters == label)[0])
        indices[-1] = np.setdiff1d(indices[-1], [i]) # remove self index
    print(len(indices))
    print(len(donnees))
    donnees['nb_voisins'] = np.array([len(neighs) for neighs in indices], dtype=np.int32)
    donnees['VID'] = [list(neighs) for neighs in indices] # voisins id

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
    clusters_uniques = np.unique(donnees['cluster'])
    clusters_uniques = [c for c in clusters_uniques if c != -1]  # Exclure les points bruyants
    nombre_clusters = len(clusters_uniques)

    for i, cluster in enumerate(clusters_uniques):
        donnees.loc[donnees['cluster'] == cluster, 'color'] = generate_color(i, nombre_clusters)

    # Export des résultats
    donnees.to_csv("res_dbscan.csv", sep=',', index=False)

def run_simulation_dbscan(dbscancsv_file):
    #1 Gbps
    run_dbscan(dbscancsv_file, 1000)
    nb_cluster_mean_1gbps = calcul_nb_clusters("res_clusters_dbscan.csv")
    nb_usrs_mean_1gbps = nb_users_par_cluster("res_clusters_dbscan.csv")
    #2 Gbps
    run_dbscan(dbscancsv_file, 2000)
    nb_cluster_mean_2gbps = calcul_nb_clusters("res_clusters_dbscan.csv")
    nb_usrs_mean_2gbps = nb_users_par_cluster("res_clusters_dbscan.csv")
    #4 Gbps
    run_dbscan(dbscancsv_file, 4000)
    nb_cluster_mean_4gbps = calcul_nb_clusters("res_clusters_dbscan.csv")
    nb_usrs_mean_4gbps = nb_users_par_cluster("res_clusters_dbscan.csv")

    # Créer un DataFrame avec les informations
    data = {
        'Débit': ['1 Gbps', '2 Gbps', '4 Gbps'],
        'Nb de clusters': [nb_cluster_mean_1gbps, nb_cluster_mean_2gbps, nb_cluster_mean_4gbps],
        'Nb d\'utilisateurs par cluster': [nb_usrs_mean_1gbps, nb_usrs_mean_2gbps, nb_usrs_mean_4gbps]
    }

    df = pd.DataFrame(data)

    # Enregistrer le DataFrame dans un fichier CSV
    df.to_csv('stats_dbscan.csv', index=False)

    print("Pour 1 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_1gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_1gbps))

    print("Pour 2 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_2gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_2gbps))

    print("Pour 4 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_4gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_4gbps))

#DBSCAN PRENDS TROP DE TEMPS - DONNEES TROP GRANDES
run_simulation_dbscan("test_small_csav_500.csv")