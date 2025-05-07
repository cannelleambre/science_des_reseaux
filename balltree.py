import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
import numpy as np
from haversine import haversine, Unit
import math


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
    donnees.to_csv("res_clusters_ball_tree.csv", sep=',', index=False)

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

def divide_cluster(donnees, cluster, max_pir):
    epsilon = 0.1
    # Divisez le cluster en sous-clusters de manière à ce que la somme des PIR dans chaque sous-cluster soit inférieure à max_pir
    sub_clusters = []
    sub_cluster_pir_sums = {}
    cluster_points = donnees.loc[donnees['cluster'] == cluster]
    while len(cluster_points) > 0:
        # Sélectionnez un point aléatoire dans le cluster
        point = cluster_points.sample(n=1)
        # Créez un nouveau sous-cluster avec le point sélectionné
        sub_cluster = [point]
        # Ajoutez les points voisins du point sélectionné au sous-cluster
        voisins = donnees.loc[(donnees['LAT'] - point['LAT']).abs() < epsilon]
        voisins = voisins.loc[(donnees['LON'] - point['LON']).abs() < epsilon]
        sub_cluster.extend(voisins)
        # Mettez à jour les somme des PIR dans le sous-cluster
        sub_cluster_pir_sum = sum(float(row) for row in sub_cluster)
        if sub_cluster_pir_sum > max_pir:
            sub_sub_clusters = divide_cluster(donnees, sub_cluster, epsilon, max_pir)
            sub_clusters.extend(sub_sub_clusters)
        else:
            sub_clusters.append(sub_cluster)
        # Retirez les points du sous-cluster du cluster original
        cluster_points = cluster_points.drop(sub_cluster.index)
    return sub_clusters

def run_dbscan(csv_file, max_pir_threshold):
    epsilon = 0.1
    # Lecture du CSV
    dtype_dict = {
        'LAT': np.float32,
        'LON': np.float32
    }
    donnees = pd.read_csv(csv_file, delimiter=",", dtype=dtype_dict)

    # Calcul des clusters avec DBSCAN
    coords = np.radians(donnees[['LAT', 'LON']].values.astype(np.float32))
    dbscan = DBSCAN(eps=epsilon, metric='haversine')
    clusters = dbscan.fit_predict(coords)

    # Assignation des clusters
    donnees['cluster'] = clusters

    # Calculez la somme des PIR des utilisateurs dans chaque cluster
    cluster_pir_sums = {}
    for cluster in np.unique(clusters):
        if cluster != -1:
            pir_sum = donnees.loc[donnees['cluster'] == cluster, 'PIR'].sum()
            cluster_pir_sums[cluster] = pir_sum

    # Vérifiez si la somme des PIR dépasse le max_pir_threshold
    for cluster, pir_sum in cluster_pir_sums.items():
        if pir_sum > max_pir_threshold:
            # Divisez le cluster en sous-clusters de manière à ce que la somme des PIR dans chaque sous-cluster soit inférieure à max_pir_threshold
            sub_clusters = divide_cluster(donnees, cluster, max_pir_threshold)            # Mettez à jour les clusters et les somme des PIR
            for sub_cluster in sub_clusters:
                donnees.loc[sub_cluster.index, 'cluster'] = len(np.unique(donnees['cluster'])) + 1

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
def run_simulation(csv_file, algorithm):
    if algorithm == "ball_tree":
        #1 Gbps
        run_ball_tree("generated.csv", 1000)
        nb_cluster_mean_1gbps = calcul_nb_clusters("res_clusters_ball_tree.csv")
        nb_usrs_mean_1gbps = nb_users_par_cluster("res_clusters_ball_tree.csv")
        #2 Gbps
        run_ball_tree("generated.csv", 2000)
        nb_cluster_mean_2gbps = calcul_nb_clusters("res_clusters_ball_tree.csv")
        nb_usrs_mean_2gbps = nb_users_par_cluster("res_clusters_ball_tree.csv")
        #4 Gbps
        run_ball_tree("generated.csv", 4000)
        nb_cluster_mean_4gbps = calcul_nb_clusters("res_clusters_ball_tree.csv")
        nb_usrs_mean_4gbps = nb_users_par_cluster("res_clusters_ball_tree.csv")

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

    elif algorithm == "DBSCAN":
        #1 Gbps
        run_dbscan("generated.csv", 1000)
        nb_cluster_mean_1gbps = calcul_nb_clusters("res_clusters_dbscan.csv")
        nb_usrs_mean_1gbps = nb_users_par_cluster("res_clusters_dbscan.csv")
        #2 Gbps
        run_dbscan("generated.csv", 2000)
        nb_cluster_mean_2gbps = calcul_nb_clusters("res_clusters_dbscan.csv")
        nb_usrs_mean_2gbps = nb_users_par_cluster("res_clusters_dbscan.csv")
        #4 Gbps
        run_dbscan("generated.csv", 4000)
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

#DBSCAN PRENDS TROP DE TEMPS - DONNEES TROP GRANDES
run_simulation("test_small_csav_500.csv", "DBSCAN")