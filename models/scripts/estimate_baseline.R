#setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## Sets your working directory to wherever this file is located
#setwd("..") ## Since this file should be in ./scripts, this moves you to the main replication directory

load("./data/metadata_master.RData")

out = aggregate(case_data$partyWinning, by=list(case_data$year), FUN=mean)

pdf("./figures/appendix_figure1.pdf")
plot(out, type="l", ylim=c(0.3, 1), xlab="Year", main="Proportion of Cases Won by the Petitioner",
     ylab="Proportion")
abline(h=mean(case_data$partyWinning), col="grey", lty=2)
lines(x=c(2000, 2015), y=rep(mean(case_data$partyWinning[case_data$year > 2000]),2), col="red", lty=2)
dev.off()
