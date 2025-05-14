#setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ## Sets your working directory to wherever this file is located
#setwd("..") ## Since this file should be in ./scripts, this moves you to the main replication directory


options(stringsAsFactors = FALSE)
library(dplyr)

dat = read.csv("./results/results_combined.csv")
scdb = read.csv("./data/SCDB_2015_01_caseCentered_Citation_trimmed.csv")
scdb$petitioner_won = scdb$partyWinning == 1

## Then do a merge and clean up scdb vars
dat = merge(dat, scdb[,c("docket", "petitioner_won", "issueArea", "petitionerState", "respondentState")])
dat$govparty = dat$petitionerState | dat$respondentState
dat$govparty[is.na(dat$govparty)] = FALSE
dat$issue2 = NA
dat$issue2[dat$issueArea == 1] = 1
dat$issue2[dat$issueArea == 2] = 2
dat$issue2[dat$issueArea == 3] = 3
dat$issue2[dat$issueArea == 8] = 8
dat$issue2[dat$issueArea == 9] = 9
dat$issue2[dat$issueArea == 10] = 10

## Then make the baseline as collapsing the petitioner_won var by year
#baseline = dat %>% group_by(term) %>% summarize(mean(petitioner_won))
## Make the kks by doing the same
#kks = dat %>% group_by(term) %>% summarize(mean(kks_correct))
#kks = c(0.66, 0.74, 0.75, 0.82, 0.77, 0.75, .79, 
#        .74,.73,.75,.69,.80,.74)


### Make baseline by collapsing petitioner_won by each of the decision margins and issue areas
kks = rbind(dat %>% group_by(decision_margin) %>% 
  summarize(kks = mean(kks_correct),cc = mean(courtcast_correct),
            mplus = mean(mplus_correct, na.rm=T), baseline = mean(petitioner_won)) %>% dplyr::select(-decision_margin),
            dat %>% filter(!is.na(issue2)) %>% group_by(issue2) %>% 
  summarize(kks = mean(kks_correct),cc = mean(courtcast_correct),
            mplus = mean(mplus_correct,na.rm=T), baseline = mean(petitioner_won))  %>% dplyr::select(-issue2),
            dat %>% group_by(govparty) %>% 
  summarize(kks = mean(kks_correct),cc = mean(courtcast_correct),
            mplus = mean(mplus_correct, na.rm=T), baseline = mean(petitioner_won))  %>% dplyr::select(-govparty))
kks = kks[-1,]

rownames = c("Margin: 5-4",
             "Margin: 6-3",
             "Margin: 7-2",
             "Margin: 8-1",
             "Margin: 9-0",
             "Issue: Criminal Procedure",
             "Issue: Civil Rights",
             "Issue: First Amendment",
             "Issue: Economic Activity",
             "Issue: Judicial Power",
             "Issue: Federalism",
             "Government is Party",
             "Government is not Party")


dmtable = data.frame(CaseType = rownames, Baseline = kks[,4], Katz = kks[,3], CourtCast = kks[,2], KKS = kks[,1] )

library(xtable)
print(xtable(dmtable), include.rownames=FALSE, file="./tables/table2.tex")





options(stringsAsFactors = FALSE)
data = read.csv("./data/scotus_results.csv", sep="\t")
colnames(data) = c("Model", "Data", "Accuracy")

baseline = data.frame(Model = rep("Baseline", 3), Data = c("scdb", "oa", "both"), Accuracy = rep(0.68, 3))

data = rbind(data, baseline)


data$Model = rep(c("KKS", "Marshall+", "CourtCast", "RandomForest", "Baseline"), each=3)
data$Data = rep(c("SCDB", "Oral Argument", "Both"), 5)
data$Data = factor(data$Data, levels=c("SCDB", "Oral Argument", "Both"))
data$Model = factor(data$Model)

data2 = data

png("./figures/figure1.png", width=1023, height=845, units="px")
interaction.plot(x.factor = data2$Data, data2$Model, data2$Accuracy, trace.label = NULL, ylim = c(0.5, 0.8),
                 col=c("darkgrey", "black", "black", "black", "black"), lwd=3, xtick=F,
                 xlab = "Training Data Set", ylab="10-fold Cross-Validation Accuracy", lty=c(1,3,4,5,2),
                 legend=F)
grid()
legend("bottomright", levels(data$Model),lty=c(1,3,4,5,2), lwd=2, col=c("darkgrey", "black", "black", "black", "black"))
points(x=c(1,2), y=c(data2$Accuracy[data2$Data == "SCDB" & data2$Model == "Marshall+"],
                     data2$Accuracy[data2$Data == "Oral Argument" & data2$Model == "CourtCast"]),
       cex=1.5, pch=16, bg=c("blue", "green"))
dev.off()