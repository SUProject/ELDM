# -*-coding:Latin-1 -*

#########################################################
# Main
# ELTDM project
# WORKS ONLY on windows
#########################################################

#####
# packages import
#####

## Common
import os
import csv
import numpy as np
import time
import multiprocessing as mltp

## Perso
import fileManagement.function as fmf
import KMeans.monoKMeans.function as monokmf
import random as rd
import KMeans.multiKMeans.function as multikmf

if __name__ == "__main__":
    
    ## seed
    rd.seed(8)

    #####
    # input import
    #####

    # folder with the input file. we put it as working directory as we write the output in it.
    inOutFolder = input("Enter working directory (currently '../input/' please): ")
    print(inOutFolder)
    os.chdir(inOutFolder)

    # number of clusters and method
    k = input("Enter number of clusters (only positive integers please): ")
    k = int(k)
    method = input("Enter distribution method (mono or multi please): ")

    # building a numpy array x from csv file
    (x, varName, inFileRow) = fmf.featureArrayFromCsv("data.csv")

#####
# clustering mono thread
#####
    if method == "mono":
        start = time.time()

        center = monokmf.initialize(x, k)
        hasConverged = False

        while not hasConverged:
            centerOld = center
            matDist = monokmf.allDistance(x, center)
            vecAlloc = monokmf.alloc(matDist)
            center = monokmf.newCenter(x, vecAlloc, k)
            hasConverged = (center == centerOld).all()

        timeElapsed = open("timeElapsed" + method + ".txt", "w")
        timeElapsed.write(str(time.time()-start))
        timeElapsed.close()
        os.system("pause")
    
    #####
    # clustering multi thread
    #####
    elif method == "multi":
        start = time.time()

        # count number of cpu
        #m = mltp.cpu_count()
        m=4
        # launch as much workers than cpu
        pool = mltp.Pool(processes=m)
        # Partitionne les données une fois multikmf.
        chunkedX = multikmf.chunk(x, m)
        # Initialiser les centres globaux
        center = monokmf.initialize(x, k)
        #print(len(center))
        # Créer la liste pour pool.map
        
        hasConverged = False
        while not hasConverged:
            centerOld = center
            xMapList = multikmf.listToMap(chunkedX, centerOld)
        
        # pool.map prend en argument une fonction et une liste
        # Role ourMap, associer un centre à chaque point: centre le plus proche du point
        # La liste est de longueur égale au nombre de process et contient les différentes
        # parties des données de départ. Chaque élément de la liste inclut les centres actuels.
        # Elle renvoit une liste avec chaque point, et le centre dont il est le plus proche
            xMapListProcessed = pool.map(multikmf.ourMap, xMapList)
        # La fonction ourReduce prend en argument la liste de réponses, constituées
        # pour chaque élément: des centres locaux dans le chunk, et leur nombre de 
        # points associés.
        # Si on voulait reparalléliser, on pourrait créer un dico dont les clefs
        # sont les centres, et les valeurs, les points associés, sans se soucier
        # du nombre de points associées, puisqu'il serait inclus dans l'information
        # "points associés".
        # Dans tous les cas elle renvoit les nouveaux centres.
            center = multikmf.ourReduce(xMapListProcessed)
            print(center)
            hasConverged = (center == centerOld).all()
        
        matDist = monokmf.allDistance(x, center)
        vecAlloc = monokmf.alloc(matDist)
        
        timeElapsed = open("timeElapsed" + method + "with" + str(m) + "processes.txt", "w")
        timeElapsed.write(str(time.time()-start))
        timeElapsed.close()
    

### Analytiquement, on peut différencier le temps de mise 
    ### en place du threading, et le temps de calcul dans le
    ### threading
    ### Time1Thread: Tester le temps avec un thread mais 
    ### en passant par pool.map
    ### TimeSeq = Le comparer au temps sequentiel
    ### TimeStruct = Time1Thread - TimeSeq évalue le temps
    ### imposer par la mise en place de la structure de thread
    ### Time4Thread Tester le temps avec 4 thread
    ### Hypothèse: 4 Thread plus rapide que 1 Thread ?
    ### Time4Thread < Time1Thread ?
    ### Hypothèse: 4 Thread plus rapide que TimeSeq ?
    ### Time4Thread < TimeSeq ?
    ### prise en compte de timestruct si on se rend compte que
    ### finalement Time4Thread > TimeSeq
    ### Time4Thread - TimeStruct < Seq ?
    ### Evaluer si timestruct est le même pour 1 ou 4 thread 
    ### ou si 4 fois plus, ou autre

    
    else:
        print("Error: Wrong method type provided.")
        # Pause of the system
        os.system("pause")

#####
# output export (vecAlloc n'existe pas encore dans multi donc on peut pas tester si à la fin)
#####
# writing the result with a new column
    xAugmented = np.hstack((x, vecAlloc)) # argument is a real tuple => (,) inside the ().
    fmf.csvFromFeatureArrayAndClust("data_clustered_" + method + str(k) + ".csv",\
                                varName,\
                                xAugmented)

                            

