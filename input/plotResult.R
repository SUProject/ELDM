###################################
# ELADM
# Script to plot the results
###################################


#Authors: Arthur Roullier, Pierre Cordier


####### cluster analysis
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


####### Time analysis
# plot of nb clusters vs time for mono and multi and vs nb loop to converge
par(mfrow = c(1,2))
plot(c(1:10), 
     c(0.1,1.97,0.51,0.91,2.21,1.89,2.36,2.69,6.24,7.4), 
     type="l", col = 'blue', lwd =c(4), 
     xlab="# clusters", ylab = "Time (sec)", 
     xlim=c(0,11), ylim = c(0, 8),
     main = c("# clusters vs time"))
lines(c(1:10), 
      c(0.65,1.95,0.91,1.47,2.10,1.86,2.28,2.56,4.57,5.23), 
      type="l", col = 'red',lwd =c(4))
legend(0.15,7.9, c('mono', 'multi 2 workers'), 
       col=c('blue','red'),lwd =c(4))

plot(c(1:10), c(2, 27,5,7,14,10,11,11,23,25), 
     type = "h",lwd="5", col = 'yellow', 
     xlab="# clusters", ylab = "# Iterations", 
     xlim=c(0,11), ylim = c(0, 28),
     main = c("# clusters vs # iterations"))

# plot # workers vs time while 2 clusters (many iterations)
# plot # workers vs time for 3 clusters (few iterations)
par(mfrow = c(1,2))
plot(c(1:10), 
     c(2.51,1.90,2.21,2.35,2.71,3.25,3.72,3.73,4.23,4.19), 
     type="l", col = 'blue', lwd =c(4), 
     xlab="# workers", ylab = "Time (sec)", 
     xlim=c(0,11), ylim = c(0.5, 5),
     main = c("Many iterations
              2 clusters"))

plot(c(1:10), 
     c(1.15,0.91,1.11,1.35,1.64,2.03,1.76,2.58,3.22,2.07), 
     type="l", col = 'red', lwd =c(4), 
     xlab="# workers", ylab = "Time (sec)", 
     xlim=c(0,11), ylim = c(0.5, 5),
     main = c("Few iterations
              3 clusters"))


