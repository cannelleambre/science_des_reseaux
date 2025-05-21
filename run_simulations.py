from balltree import run_ball_tree
from dbscan import run_dbscan
from calculs_clusters_stats import calcul_nb_clusters_ball_tree, nb_users_par_cluster_ball_tree
import pandas as pd

csv_file = "generated.csv"


algorithm = input("Quel algorithme voulez-vous utiliser ? (ball_tree / dbscan) : ")
if algorithm == 'ball_tree':
    #1 Gbps
    run_ball_tree(csv_file, 1000)
    nb_cluster_mean_1gbps = calcul_nb_clusters_ball_tree("res/ball_tree/res_clusters_" + algorithm + "_1000" + ".csv")
    nb_usrs_mean_1gbps = nb_users_par_cluster_ball_tree("res/ball_tree/res_clusters_" + algorithm + "_1000" + ".csv")
    #2 Gbps
    run_ball_tree(csv_file, 2000)
    nb_cluster_mean_2gbps = calcul_nb_clusters_ball_tree("res/ball_tree/res_clusters_" + algorithm + "_2000" + ".csv")
    nb_usrs_mean_2gbps = nb_users_par_cluster_ball_tree("res/ball_tree/res_clusters_" + algorithm + "_2000" + ".csv")
    #4 Gbps
    run_ball_tree(csv_file, 4000)
    nb_cluster_mean_4gbps = calcul_nb_clusters_ball_tree("res/ball_tree/res_clusters_" + algorithm + "_4000" + ".csv")
    nb_usrs_mean_4gbps = nb_users_par_cluster_ball_tree("res/ball_tree/res_clusters_" + algorithm + "_4000" + ".csv")

    # Créer un DataFrame avec les informations
    data = {
        'Débit': ['1 Gbps', '2 Gbps', '4 Gbps'],
        'Nb de clusters': [nb_cluster_mean_1gbps, nb_cluster_mean_2gbps, nb_cluster_mean_4gbps],
        'Nb d\'utilisateurs par cluster': [nb_usrs_mean_1gbps, nb_usrs_mean_2gbps, nb_usrs_mean_4gbps]
    }

    df = pd.DataFrame(data)

    # Enregistrer le DataFrame dans un fichier CSV
    df.to_csv('stats/stats_' + algorithm + '.csv', index=False)

    print("Pour 1 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_1gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_1gbps))

    print("Pour 2 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_2gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_2gbps))

    print("Pour 4 Gbps :")
    print("Nombre de clusters :" + str(nb_cluster_mean_4gbps))
    print("Nombre moyen de terminaux par clusters :" + str(nb_usrs_mean_4gbps))

elif algorithm == 'dbscan':
    #1 Gbps
    
    stats_1000 = run_dbscan(csv_file, 1000)
    stats_2000 = run_dbscan(csv_file, 2000)
    stats_4000 =run_dbscan(csv_file, 4000)
    stats_globales = pd.concat([stats_1000, stats_2000, stats_4000])
    print(stats_globales)
    stats_globales.to_csv('stats/stats_' + algorithm + '.csv', index=False)
else:
    print("Algorithme inconnu")