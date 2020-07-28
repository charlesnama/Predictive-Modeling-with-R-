
rm(list = ls(all=TRUE))

#Get data 
# Preprocessing
#Build model 
#Plot ROC
#Confusion matrix 


bank <- read.table(file ='/Users/mac/Desktop/bank.txt', header = T, sep = ';')
summary(bank)
str(bank)

sum(is.na(bank))
set.seed(123)

library(caret)  ###Library containing all the metrics 
trainrows <- createDataPartition(bank$y, p=0.7, list = F)
train_data <- bank[trainrows,]
test_data <- bank[-trainrows,]
dim(trainrows)
dim(test_data)
dim(train_data)

##building the model with generalised linear model function glm 



log_reg <- glm(y~.,data = train_data, family = 'binomial')## one dependent variable y~Age for example/ or y~. for more 
summary(log_reg)
#Signif. codes:  0 ‘***’ 0.001 (99.9%) ‘**’ 0.01 (99%)‘*’ 0.05 (95%)‘.’ 0.1 (90%)‘ ’ 1 
###Check for running multiple logit models to have different AIC  
###Check for write output to csv 

##ROC plot 

prob_train <- predict(log_reg,train_data, type = "response") ##type = response means return probability not class taking .5 as 1/0
head(prob_train)

install.packages("ROCR")
library(ROCR)  ###to calculate confusion matrix 

pred <- prediction(prob_train,train_data$y)
pred
##prediction as probabilities 
##labels yes/no 
##cutoffs
##fp/tp/tn/fn
#number of negative pos/ number of pos pred

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf,colorize=T, print.cutoffs.at=seq(0,1,0.05))

perf_auc <- performance(pred, measure = "auc") ##performance function also has auc 
auc <- perf_auc@y.values[[1]] ##this will get the auc 
print(auc)

###chosing the cutoff value based on the ROC curve view 


prob_test <- predict(log_reg, test_data, type="response")
pred_test <- ifelse(prob_test > 0.1, "yes","no")

#summary - build model on train (glm), predicted on train (predict), yhat and actual(prediction), then move
##to (performance), then move to (roc and auc)
print(pred_test) ##print predicted 
install.packages("e1071")
library(e1071)
confusionMatrix(as.factor(pred_test), test_data$y, positive = "yes") ##predicted value and actual values

#intepreting confusion matrix check for more of this and how to determine good model 

 
