import pandas as pd
import matplotlib.pyplot as plt

nom_CSV = input("Nom du fichier CSV : ")

# Chargement du fichier CSV
data = pd.read_csv(nom_CSV + ".csv")

# Récupération des identifiants des clusters
cluster_ids = data['cluster'].unique()
nb_clusters = len(cluster_ids)

# Attribution d'une couleur à chaque cluster
colors = plt.cm.get_cmap('tab20', nb_clusters)
clusters_colored = {}
for i in range(nb_clusters):
    clusters_colored[cluster_ids[i]] = colors(i)[:3]

# Conversion des couleurs RGB en hexadécimal
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(int(255 * c) for c in rgb)

# Ajout de la colonne 'cluster_color' au CSV
data['cluster_color'] = data['cluster'].map(lambda cid: rgb_to_hex(clusters_colored[cid]))

# Création d'un nouveau CSV avec les clusters colorés
data.to_csv("colored_" + nom_CSV + ".csv", index=False)