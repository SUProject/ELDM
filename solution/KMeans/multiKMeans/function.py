# -*-coding:Latin-1 -*

#########################################################
# Test functions
# ELADM Project
#########################################################

#####
# packages import
#####
# common
import os
import multiprocessing as mltp
import numpy as np
# perso
import monoKMeans.function as monokmf

#####
# Modulus part for import in the main
#####

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
        global center
        
        for xMap in listChunk:

                ### Computation of the distances between points and centers
                matDistMap = monokmf.allDistance(xMap, center)
                ### Computation of the local allocation
                vecAllocMap = monokmf.alloc(matDistMap)
                ### Computation of local centers and populations
                (centerMap, nbMap) = newCenterMap(xMap, vecAllocMap, k)

                res.append((centerMap, nbMap))

        return(res)

def ourReduce(listMapped):
        """
        Function that computes the new real global centers
        according to the result of ourMap().
        """
        # initialization on the first local center, for the shape
        (k, p) = listMapped[0][0].shape
        
        center = np.zeros((k,p))
        m = len(listMapped)
        
        # global center
        for ic in range(0, k):
                nbPoint = 0
                # number of points around center k
                for il in range(0, m):
                        nbPoint = nbPoint + listMapped[il][1][ic]
                # weighted average
                for il in range(0, m):
                        center[ic,:] = center[ic,:] + \
                                 1/nbPoint * listMapped[il][1][ic] * listMapped[il][0][ic]

        return(center)



#####
# Modulus part for test here
#####

if __name__ == "__main__":
        
        ### object to be chunked
        k = 2
        m = 4
        n = 9
        x = np.array([[1,2,3],\
                      [11,12,13],\
                      [21,22,23],\
                      [31,32,33],\
                      [41,42,43],\
                      [51,52,53],\
                      [61,62,63],\
                      [71,72,73],\
                      [81,82,83]])
        # center is kept as a global variable if it works...
        center = np.array([[10,10,10],\
                      [50,50,50]])
              
##        ### Build a pool of m processes
##        pool = mltp.Pool(processes = m,)
##   
        # pool = mltp.Pool(processes = 2)
        ### Fragment the string data into m chunks
        xMapList = chunk(x, m)
        print(xMapList)
        
        ### Generate local centers
        ### (and number of points by local center, needed for the final center)
        xMapListProcessed = ourMap(xMapList)
        print("yo le resultat !")
        print(xMapListProcessed)
        print(xMapListProcessed[0][0].shape)
        print(xMapListProcessed[0][0][1])

##        single_count_tuples = pool.map(Map, partitioned_text)
        os.system("pause")
 
        ### Organize the count tuples; lists of tuples by token key
        ##token_to_tuples = Partition(single_count_tuples)
        ##

        
        ### Collapse the lists of results into global centers
        center2 = ourReduce(xMapListProcessed)
        print(center2)
        
        os.system("pause")