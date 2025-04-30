import csv
#mport pandas as pd
import math

nb_ligne = 0
filename = "generated.csv"
with open(filename) as csvfile:
    for ligne in csvfile:
        if nb_ligne <3:      
            print(ligne)

        nb_ligne +=1

def calcul_distance(longitude1, latitude1, longitude2, latitude2):
    longitude = abs(longitude1 - longitude2)
    latitude = abs(latitude1 - latitude2)
    distance = math.sqrt(longitude**2 + latitude**2)
    return distance

#update csv cluster : nb_user +1, debit_restant -debit_user
def ajout_user(id, centre, debit_user):
    filename = "clusters.csv"
    with open(filename) as csvfile:
        writer = csv.writer(csvfile)
        for ligne in csvfile:
            if(ligne[4]==centre & ligne[0]==id):
                #ligne du cluster
                ligne[1] +=1
                ligne[2] -= debit_user
            else:
                #il n'existe pas de cluster
                new_row = []
                writer.writerow(column_name)



#RAFFINAGE
#lit ligne par ligne le fichier data
#pour chaque ligne courante
    # liste les clusters voisins
        # barycentre < 45km
        # garde cluster debit dispo 
            # si pls cluster > debit plus dispo
        
    # ajoute terminal au cluster
        # maj débit
        # maj nb user
    # si pas de cluster dispo
        #on en créée un de centre lui-même

# stocker clusters ? csv?
# id;nb_user;capatité_restante;centre
