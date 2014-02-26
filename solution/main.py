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
        i=0
        
        while not hasConverged:
            i += 1
            centerOld = center
            matDist = monokmf.allDistance(x, center)
            vecAlloc = monokmf.alloc(matDist)
            center = monokmf.newCenter(x, vecAlloc, k)
            hasConverged = (center == centerOld).all()
        
        # writing the time
        timeElapsed = open("time/timeElapsed" + method + str(k) + "clusters.txt", "w")
        timeElapsed.write(str(time.time()-start) + "\n" + str(i))
        timeElapsed.close()
        # System break
        os.system("pause")
    
    #####
    # clustering multi thread
    #####
    elif method == "multi":
        start = time.time()

        # count number of cpu
        m = mltp.cpu_count()
        # launch as many workers as # cpu
        pool = mltp.Pool(processes=m)
        # Partitioning the data
        chunkedX = multikmf.chunk(x, m)
        # Initialize the centers
        center = monokmf.initialize(x, k)
        
        i = 0     
        hasConverged = False
        while not hasConverged:
            i += 1
            centerOld = center
            # Creation of the list for pool.map
            xMapList = multikmf.listToMap(chunkedX, centerOld)
            # Mapping of the chunks
            xMapListProcessed = pool.map(multikmf.ourMap, xMapList)
            # weighted average of the centers among the chunks
            center = multikmf.ourReduce(xMapListProcessed)
            hasConverged = (center == centerOld).all()
        
        # Final allocation vector
        matDist = monokmf.allDistance(x, center)
        vecAlloc = monokmf.alloc(matDist)
        
        timeElapsed = open("time/timeElapsed" + method + str(k) + "cluster" + str(m) + "process.txt", "w")
        timeElapsed.write(str(time.time()-start))
        timeElapsed.close()
        
        # No system break, to get relevant measure of time elapsed.
        
    else:
        print("Error: Wrong method type provided.")
        # System break
        os.system("pause")

#####
# output export (vecAlloc n'existe pas encore dans multi donc on peut pas tester si à la fin)
#####
# writing the result with a new column
    xAugmented = np.hstack((x, vecAlloc)) # argument is a real tuple => (,) inside the ().
    fmf.csvFromFeatureArrayAndClust("output/data_clustered_" + method + str(k) + ".csv",\
                                varName,\
                                xAugmented)

                            

