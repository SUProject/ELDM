# -*-coding:Latin-1 -*

#########################################################
# Main commentaire secret
# ELADM project
# WORKS ONLY on windows
#########################################################
# import other files:
# http://fr.openclassrooms.com/informatique/cours/apprenez-a-programmer-en-python/je-viens-pour-conquerir-le-monde-et-creer-mes-propres-modules
# warning on copy of objects and modifications on them
# http://fr.openclassrooms.com/informatique/cours/apprenez-a-programmer-en-python/la-portee-des-variables-2
#numpy unofficial for 64 bits
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
# tutorial numpy
# http://wiki.scipy.org/Tentative_NumPy_Tutorial#head-c5f4ceae0ab4b1313de41aba9104d0d7648e35cc
#########################################################
#
#####
# packages import & seed
#####

## Common
import os
# import pickle # to save / load any object (not used here)
import csv
import numpy as np
import time
import multiprocessing as mltp

## Perso
import fileManagement.function as fmf
import KMeans.monoKMeans.function as monokmf
import random as rd
import KMeans.multiKMeans.function as multikmf

## seed
rd.seed(8)

#####
# input import
#####

# folder with the input file. we put it as working directory as we write the output in it.
inOutFolder = input("Enter working directory (currently ../input/ please): ")
print(inOutFolder)
os.chdir(inOutFolder)

# number of clusters and method
k = input("Enter number of clusters (only positive integers please): ")
k = int(k)
method = input("Enter distribution method (mono or multi please): ")

# building a dictionary x from csv file
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

    end = time.time()
    print("""Execution time of sequential K-means = """, end - start)

    #####
    # output export (vecAlloc n'existe pas encore dans multi donc on peut pas tester si à la fin)
    #####
    # writing the result with a new column
    xAugmented = np.hstack((x, vecAlloc)) # argument is a real tuple => (,) inside the ().
    fmf.csvFromFeatureArrayAndClust("data_clustered_" + method + str(k) + ".csv",\
                               varName,\
                               xAugmented)

    
    
#####
# clustering multi thread
#####
elif method == "multi":
    
    
    start = time.time()

    # count number of cpu
    m = mltp.cpu_count()
	# lance autant de workers qu'il y a de cpu
    pool = mltp.Pool(processes=m)    
    # Partitionne les données une fois
    xMapList = multikmf.chunk(x, m)
    # Initialiser les centres globaux
    center = monokmf.initialize(x, k)
    #hasConverged = False

    #while not hasConverged:
    #    centerOld = center
    #    xMapListProcessed = pool.map(mutlikmf.ourMap, xMapList)
    #    center = pool.map(multikmf.ourReduce, xMapListProcessed)
    #    hasConverged = (center == centerOld).all()
    
    #####
    # output export (vecAlloc n'existe pas encore dans multi donc on peut pas tester si à la fin)
    #####
    # writing the result with a new column
    #xAugmented = np.hstack((x, vecAllocMap)) # argument is a real tuple => (,) inside the ().
    #fmf.csvFromFeatureArrayAndClust("data_clustered_" + method + str(k) + ".csv",\
    #                           varName,\
    #                           xAugmented)

    
    end = time.time()
    print("""Execution time of mutli-processed K-means = """, end - start)

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

          

#####
# test zone
#####



# test of bissextile() in sequentialKmeans
#monokmtf.bissextile()

# Pause of the system
os.system("pause")

