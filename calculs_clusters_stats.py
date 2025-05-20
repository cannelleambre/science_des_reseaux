import pandas as pd
from collections import defaultdict
import csv
def calcul_nb_clusters_ball_tree(csv_file):

    # Charger le fichier CSV
    df = pd.read_csv(csv_file)

    # Compter le nombre de clusters uniques
    num_clusters = df['cluster'].nunique()

    return(num_clusters )

def nb_users_par_cluster_ball_tree(fichier):
    # Charger le fichier CSV
    df = pd.read_csv(fichier)

    # Compter le nombre d'utilisateurs par cluster
    nb_users_par_cluster = df['cluster'].value_counts()

    # Calculer la moyenne du nombre d'utilisateurs par cluster
    moyenne_nb_users = nb_users_par_cluster.mean()

    return( moyenne_nb_users)


def nb_users_par_cluster_dbscan(fichier_csv):
    # Dictionnaire pour compter le nombre d'utilisateurs par cluster
    utilisateurs_par_cluster = defaultdict(int)

    # Lire le fichier CSV
    with open(fichier_csv, mode='r') as fichier:
        lecteur_csv = csv.reader(fichier)
        next(lecteur_csv)  # Sauter l'en-tête si nécessaire

        for ligne in lecteur_csv:
            cluster = int(ligne[3])  # Supposons que le cluster est dans la quatrième colonne
            utilisateurs_par_cluster[cluster] += 1

    # Calculer le nombre moyen d'utilisateurs par cluster
    nombre_clusters = len(utilisateurs_par_cluster)
    nombre_total_utilisateurs = sum(utilisateurs_par_cluster.values())
    nombre_moyen_utilisateurs = nombre_total_utilisateurs / nombre_clusters

    return nombre_moyen_utilisateurs
