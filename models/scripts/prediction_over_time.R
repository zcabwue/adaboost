#setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## Sets your working directory to wherever this file is located
#setwd("..") ## Since this file should be in ./scripts, this moves you to the main replication directory

dat = rbind(c(2013, .6798),
            c(2014, .697),
            c(2015, .700),
            c(2016, .702),
            c(2017, .7404))

png("./figures/appendix_figure2.png", height=878, width=1024, units="px")
plot(dat, type="b", xlim = c(2013, 2017), ylim=c(.66, .76),
     xlab="Year", ylab = "Predictive Accuracy", main="Predictive Accuracy of the Best Model")
text(x=2013.15, y=.675, "Baseline", col="red")
text(x=2014, y=.705, "{Marshall}+", col="red")
text(x=2015, y=.705, "Courtcast", col="red")
text(x=2016, y=.695, "Katz et al", col="red")
text(x=2016.85, y=.745, "KKS", col="red")
dev.off()