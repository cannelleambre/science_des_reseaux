import pandas as pd
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
    # Lire le fichier CSV
    df = pd.read_csv(fichier_csv)
    
    # Calculer la moyenne du nombre d'utilisateurs par cluster
    moyenne = df.groupby('cluster')['PIR'].mean()
    
    return moyenne