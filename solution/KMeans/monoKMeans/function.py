# -*-coding:Latin-1 -*

#########################################################
# Functions for mono KMeans
# ELADM Project
#########################################################

#####
# Package import
#####

## Common
import numpy as np
import random as rd

#####
# Modulus part for import in the main
#####

def initialize(x, k):
        """
        Creates an array of centers
        from k random points in the input data.
        """
        n = x.shape[0]
        s = rd.sample(range(0,n), k)

        center = np.array(x[s,:])

        return(center)


def distance(u, v, kernel = "linear"):
        """
        Returns the distance between two vectors u, v.
        u and v should be arrays.
        """
        if kernel == "linear":
                d2 = (u**2).sum() + (v**2).sum() - 2 * (u*v).sum()

        return(d2)

        
def allDistance(x, center, kernel = "linear"):
        """
        Returns all the distances between the points in x
        and those in center.
        This function should be parallelized
        """
        n = x.shape[0]
        k = center.shape[0]
        matDist = np.zeros((n, k))

        for ix in range(0, n): # here would become n/4...
                for ic in range(0, k):
                        matDist[ix,ic] = distance(x[ix,:], center[ic,:], kernel)

        return(matDist)


def alloc(matDist):
        """
        Returns a (n,1) vector with the number of
        of the closest center for each point in x.
        """
        n = matDist.shape[0]
        vecAlloc = np.zeros((n, 1))
        for i in range(0, n):
                w = np.where(matDist[i,:] == matDist[i,:].min())[0] # [0] because where returns a tuple (array, )
                vecAlloc[i] = w[0] # [0] because we want only one if numerous

        return(np.array(vecAlloc, dtype = int))


def newCenter(x, vecAlloc, k):
        """
        Returns the updates centers with regard to
        the new allocation of x's arround the new centers.
        WARNING : we don not take into account the case when
        a center 'has no point around it'. Needs reflexion.
        this functions should be mapped for local centers
        and reduced for aggregation.
        """
        p = x.shape[1]
        
        kNew = np.unique(vecAlloc).shape[0]
        if kNew < k:
                print("Error (lack of developpemet):\
                the number of clusters has droped as at least one center has no neighbor.")                

        center = np.zeros((k, p))
        for ic in range(0, k):
                w = np.where(vecAlloc[:,0] == ic)[0] # [0] because where returns a tuple (array, )
                center[ic,:] = x[w,:].mean(axis = 0)

        return(center)



#####
# Modulus part for test here
#####
if __name__ == "__main__":
        print("t1")
        k = 2
        x = np.array([[1,2,3],[4,5,6],[7,8,9]])
        c = initialize(x, k)
        print(c)
        print("t2")
        print(distance(np.array([1,-1]), np.array([1,1])))
        print("t3")
        matDist = allDistance(x, c)
        print(matDist)
        print("t4")
        v = alloc(matDist)
        print(v)
        print("t5")
        cc = newCenter(x, v, k)
        print(cc)
