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
#import KMeans.multiKMeans.function as multikmf



#def function qui iront dans multiKmeans

# pool.map prend en argument une fonction et une liste. Les éléments de la liste
# sont callable par la fonction. pool est un objet constitué de plusieurs processus.
# map est une procédure callable sur les objets de ce type. Dans chaque processus de pool, 
# map met en mémoire la fonction, et un certain nombre déléments de la liste; tel que chaque 
# élémment soit réparti vers l'un des processus, sans répétition et sans oubli. Dans
# chaque processus, map demande l'appel de la fonction sur chaque élément qui s'y trouve.
# La sortie est la liste des réponses de la fonction à chaque élément. Si la fonction
# renvoit un dictionnaire, la sortie est une liste de dictionnaire. La liste de sortie
# est de la même taille que la liste d'entrée.

def chunk(x, m):
        """
        Returns a list() of m subsets of the array x by row
        i.e. the subsets have the same number of columns.
        """
        n = x.shape[0]
        quo, mod = divmod(n, m)
        res = []
        for i in range(0, m):
                if i < m-1:
                        res.append(x[(i*quo):((i+1)*quo),:])
                else:
                        res.append(x[(i*quo):((i+1)*quo + mod),:])
        return(res)

def listToMap(myChunk, myCenters):
        """
        Returns a list. Each element a piece of the chunk + centers.
        """
        res = []
        for chunk in myChunk:
                res.append([chunk, myCenters])
        return(res)
    
    
def newCenterMap(xMap, vecAllocMap, k):
        """
        Works on a CHUNKED input -> needs to be encapsuled in ourMap(L)
        
        Returns the LOCAL updates centers with regard to
        the new allocation of x's arround the new centers.
        WARNING : we don't take into account the case when
        a center 'has no point around it'.
        This function also returns the number of points
        arround each local center as it is needed
        to compute the final centers.
        """
        p = xMap.shape[1]
        
        kNew = np.unique(vecAllocMap).shape[0]
        if kNew < k:
                print("Warning: a center at least has no point around it.")

        centerMap = np.ones((k, p))
        nbMap = np.zeros((k,1))
        for ic in range(0, k):
                w = np.where(vecAllocMap[:,0] == ic)[0] # [0] because where returns a tuple (array, )
                nbMap[ic,0] = int(w.shape[0])
                if nbMap[ic,0] != 0:
                        # else we let the initialzing "ones", as they will not be used since nbMap = 0
                        centerMap[ic,:] = xMap[w,:].mean(axis = 0)
        return(centerMap, nbMap)


def ourMap(listChunk):
        """
        Function that maps what we want to all the
        elements of a list. The list is typically
        the list of chunks of our input data.

        The (real global) centers are supposed to be known.
        """
        res = []
        center = listChunk[1]
        xMap = listChunk[0]
        ### Computation of the distances between points and centers
        matDistMap = monokmf.allDistance(xMap, center)
        ### Computation of the local allocation
        vecAllocMap = monokmf.alloc(matDistMap)
        ### Computation of local centers and populations
        (centerMap, nbMap) = newCenterMap(xMap, vecAllocMap, len(center))

        res.append((centerMap, nbMap))
        return(res)
    
if __name__ == "__main__":
    ## seed
    rd.seed(8)

    #####
    # input import
    #####

    # folder with the input file. we put it as working directory as we write the output in it.
    inOutFolder = "../input/    "#input("Enter working directory (currently ../input/ please): ")
    print(inOutFolder)
    os.chdir(inOutFolder)

    # number of clusters and method
    global k
    k = 3   #input("Enter number of clusters (only positive integers please): ")
    k = int(k)
    method = "multi" #input("Enter distribution method (mono or multi please): ")

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
        # Partitionne les données une fois multikmf.
        chunkedX = chunk(x, m)
        # Initialiser les centres globaux
        center = monokmf.initialize(x, k)
        #print(len(center))
        # Créer la liste pour pool.map
        xMapList = listToMap(chunkedX, center)
        
        # pool.map prend en argument une fonction et une liste
        # Role ourMap, associer un centre à chaque point: centre le plus proche du point
        # La liste est de longueur égale au nombre de process et contient les différentes
        # parties des données de départ. Chaque élément de la liste inclut les centres actuels.
        # Elle renvoit une liste avec chaque point, et le centre dont il est le plus proche
        xMapListProcessed = pool.map(ourMap, xMapList)
        print(xMapListProcessed)
        # La fonction ourReduce prend en argument un dictionnaire, dont les clefs
        # sont les centres, et les valeurs, les points associés + le nombre de points associés.
        # Elle renvoit points et leur centre associé.
    
    #while not hasConverged:
    #    centerOld = center
    #xMapListProcessed = pool.map(mutlikmf.ourMap, xMapList)
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

