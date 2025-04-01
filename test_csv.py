import csv

nb_ligne = 0
filename = "generated.csv"
with open(filename) as csvfile:
    for ligne in csvfile:
        if nb_ligne <4:      
            mot = ligne.split(",")  
            print(mot)

        nb_ligne +=1

