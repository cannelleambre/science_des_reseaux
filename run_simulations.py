from balltree import run_ball_tree
from dbscan import run_dbscan
from calculs_clusters_stats import calcul_nb_clusters, nb_users_par_cluster
import pandas as pd
csv_file = "test_small_csav_500.csv"
algorithms = {
    'ball_tree': run_ball_tree,
    'dbscan': run_dbscan
}

algorithm = input("Quel algorithme voulez-vous utiliser ? (ball_tree / dbscan) : ")
func = algorithms.get(algorithm)
if func:
    #1 Gbps
    func(csv_file, 1000)
    nb_cluster_mean_1gbps = calcul_nb_clusters("res/res_clusters_" + algorithm + ".csv")
    nb_usrs_mean_1gbps = nb_users_par_cluster("res/res_clusters_ball_tree.csv")
    #2 Gbps
    func(csv_file, 2000)
    nb_cluster_mean_2gbps = calcul_nb_clusters("res/res_clusters_" + algorithm + ".csv")
    nb_usrs_mean_2gbps = nb_users_par_cluster("res/res_clusters_" + algorithm + ".csv")
    #4 Gbps
    func(csv_file, 4000)
    nb_cluster_mean_4gbps = calcul_nb_clusters("res/res_clusters_" + algorithm + ".csv")
    nb_usrs_mean_4gbps = nb_users_par_cluster("res/res_clusters_" + algorithm + ".csv")

    # Créer un DataFrame avec les informations
    data = {
        'Débit': ['1 Gbps', '2 Gbps', '4 Gbps'],
        'Nb de clusters': [nb_cluster_mean_1gbps, nb_cluster_mean_2gbps, nb_cluster_mean_4gbps],
        'Nb d\'utilisateurs par cluster': [nb_usrs_mean_1gbps, nb_usrs_mean_2gbps, nb_usrs_mean_4gbps]
    }

    df = pd.DataFrame(data)

    # Enregistrer le DataFrame dans un fichier CSV
    df.to_csv('stats_' + algorithm + '.csv', index=False)

    print("Pour 1 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_1gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_1gbps))

    print("Pour 2 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_2gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_2gbps))

    print("Pour 4 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_4gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_4gbps))

else:
    print("Algorithme inconnu")