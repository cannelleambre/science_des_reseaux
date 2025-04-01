# Sujet 2 : Clustering de terminaux utilisateurs dans un réseau de constellations de satellites
Ce projet vise à construire des clusters d’utilisateurs qui serviront à pointer les faisceaux d’une constellation de satellites à orbites basses. Les satellites orientent leurs antennes de façon à garder l’empreinte des satellites fixe au sol.

Pour l’instant la colonne du type de service n’est pas à considérer.
Voici les hypothèses et le problème posé :
    • Cluster de taille fixe : 90 km de diamètre
    • Pas de contrainte sur le centre de cluster
    • Plusieurs clusters peuvent se superposer totalement (même centre de cluster) ou partiellement
    • Débit max par cluster, 3 cas à traiter : 1Gbps, 2Gbps et 4Gbps
Objectif : Essayer de minimiser le nombre de clusters en regroupant les utilisateurs dans des clusters qui respectent les contraintes mentionnées ci-dessus.
Deux choix sont à faire quant à l’algorithme de clustering : 
Ordre de traitement : en premier lieu l’ordre dans lequel les points sont traités (par demande, aléatoire, par défaut,…)
Ordre d’agrégation des points dans les clusters : les points sont ajoutés (décroissante/croissant de distance au centre, de demande,…).
Plusieurs de ces heuristiques peuvent être codées et l’algorithme de clustering peut être itératif (ou pas).
Enfin, on s’attachera à prendre du recul par rapport à l’objectif du projet. L’idée de minimiser le nombre de clusters est-elle une bonne idée ?
CIR commited information rate : minimum garanted bandwidth
PIR peak information rate
