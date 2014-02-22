# -*-coding:Latin-1 -*

#########################################################
# Functions for file management
# ELADM Project
#########################################################
# Needs the numpy module
#########################################################


#####
# packages import
#####

## Common
import numpy as np


#####
# Modulus part for import in the main
#####
def featureArrayFromCsv(filePath):
	"""
	Builds a dictionary from a csv file
	THAT STRICTLY CORRESPONDS TO THE SIMPLEST CASE
	Many improvement can be done but it works for our simple data.csv
	Also returns the names, usefull for the output.
	"""

	# input file
	inFile = open(filePath, "r") # class TextIoWrapper

	print("Reading file...")
	inFileContent = inFile.read() # class str

	# split by lines and cleaning
	inFileRow = inFileContent.split("\n")
	inFileRow.remove('')

	# variables names and features only
	varName = inFileRow[0].split(",")
	inFileRow = inFileRow[1:]

	# array from the features
	n = len(inFileRow)
	p = len(varName)
	
	x = np.zeros((n, p))
	for indRow in range(0, n):
	    s = inFileRow[indRow].split(",")
	    for indCol in range(0, p):
	        x[(indRow, indCol)] = float(s[indCol])
	
	# close the file
	inFile.close()

	return(x, varName, inFileRow)


def csvFromFeatureArrayAndClust(filePath, varName, array):
	"""
	Builds a csv file from the array given
	"""
	n = array.shape[0]

	# open the file
	outFile = open(filePath, "w") # overwrites if exists, creates if not

	# rewritting
	# first line
	varName.append("cluster")
	outFileContent = ",".join(varName)
	# rest of the file (we first change all the entries to str)
	array = np.array(array, dtype = str)
	for indRow in range(0, n):
	    s = ",".join(array[indRow, :])
	    outFileContent = outFileContent + "\n" + s

	print("Writing file...")
	outFile.write(outFileContent) # only works with str objects

	# close the file
	outFile.close()


#####
# Modulus part for test here
#####

#if __name__ == "__main__":

       
