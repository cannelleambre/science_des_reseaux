import pandas as pd 
import numpy as np
import math
from haversine import haversine, Unit
from sklearn.cluster import KMeans
from geopy.distance import great_circle


def run_kmeans(data, max_throughput) :

    # Paramètres 

    diametre = 90
    clusters_opti = None

    # Lecture du fichier CSV 

    read_csv = pd.read_csv(data)
    coords = read_csv.iloc[:, [0, 1]].values

    # Calcul du nombre minimal de clusters 
    demande_debit = read_csv.iloc[:, 2].values
    demande_total = sum(demande_debit)
    min_k = int(np.ceil(demande_total / max_throughput))

    for i in range(min_k, len(coords) + 1) :

        cluster_valide = True
        clusters = []
        kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, random_state = 0)
        labels = kmeans.fit_predict(coords)
        centres = kmeans.cluster_centers_

        for k in range(i) :

            cluster_points = coords[labels == k]
            cluster_demande = demande_debit[labels == k]

            # Vérification des contraintes de débit et diamètre

            if sum(cluster_demande) > max_throughput :
                cluster_valide = False
                break

            distance_max = 0
            for point in cluster_points :
                distance = haversine(point, centres[k])
                if distance > distance_max :
                    distance_max = distance
            if distance_max * 2 > diametre :
                cluster_valide = False
                break

            clusters.append({
                'centres' : centres[k],
                'utilisateurs' : list(zip(cluster_points, cluster_demande)),
                'débit' : sum(cluster_demande)
            })

        if cluster_valide :
            clusters_opti = clusters
            break
        
    return clusters_opti

def sauvegarde_clusters(clusters) :

    data = []
    for cluster_id, cluster in enumerate(clusters, 1) :

        for uti in cluster['utilisateurs'] :
            (lat, lon), demande = uti
            data.append({
                'id' : cluster_id,
                'Lat' : lat,
                'Lon' : lon,
                'PIR' : demande,
                'centre' : cluster['centre']
            })
    pd.DataFrame(data).to_csv("res_clusters.csv", index = False)



# 1 Gbps

cluster_1 = run_kmeans("/Users/capucinedavid/Downloads/generated.csv", 1000)
sauvegarde_clusters(cluster_1)
print(f"Nombre de clusters crées pour 1 Gbps : {len(cluster_1)}")

# 2 Gbps

cluster_2 = run_kmeans("/Users/capucinedavid/Downloads/generated.csv", 2000)
sauvegarde_clusters(cluster_2)
print(f"Nombre de clusters crées pour 2 Gbps : {len(cluster_2)}")


    


    

