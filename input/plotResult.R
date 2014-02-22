###################################
# ELADM
# Script to plot the results
###################################


#Authors: Arthur Roullier

# set directory

dataPath <- "C:/Users/supo/Documents/GitHub/ELTDM/input/"
method = "mono"
k =5

data <-  read.table(paste(dataPath,"data_clustered_", method, k, ".csv",sep=""), sep=",", header = TRUE)
data$cluster = as.integer(data$cluster)



# plot of the best set of centers (leading to lowest distorsion)
par(mfrow = c(1,1))
plot(data[, 1], data[, 2], col = 'white', xlab="", ylab = "", main = paste(method,k,sep=""))
points(data[which(data$cluster == 0),1], data[which(data$cluster == 0),2], col = 'blue', pch = 1)
points(data[which(data$cluster == 1),1], data[which(data$cluster == 1),2], col = 'red', pch = 2)
points(data[which(data$cluster == 2),1], data[which(data$cluster == 2),2], col = 'purple', pch = 3)
points(data[which(data$cluster == 3),1], data[which(data$cluster == 3),2], col = 'black', pch = 4)
points(data[which(data$cluster == 4),1], data[which(data$cluster == 4),2], col = 'green', pch = 5)

