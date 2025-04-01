import csv
import math

nb_ligne = 0
filename = "generated.csv"
with open(filename) as csvfile:
    for ligne in csvfile:
        if nb_ligne <4:      
            mot = ligne.split(",")  
            print(mot)

        nb_ligne +=1

def calcul_distance(longitude1, latitude1, longitude2, latitude2):
    longitude = abs(longitude1 - longitude2)
    latitude = abs(longitude1 - longitude2)
    distance = math.sqrt(longitude**2 + latitude**2)
    return distance

#ALGO
# 1ère methode 
# lit ligne par ligne le fichier
# sur la ligne courante, on regarde si le terminal peut appartenir à cluster voisin
    # distance < 45 km du centre
    # debit acceptable par le beam
# si oui on l'y ajoute
    # liste ?
# sinon on le créé
