"""
Problem Statement - Predict flight delays
*Process outline*
-Import 
-Preprocess
-Modeling and Evaluation with Naive Bayes
"""

flight <- read.csv(file = '/Users/mac/Desktop/The Data Science Project /data/FlightDelays.csv', header=TRUE, sep=',')
str(flight)
summary(flight)
head(flight)
tail(flight)

colSums(is.na(flight)) #0

"""Appropriate data type conversions"""
flight$Weather <- as.factor(flight$Weather)
flight$DAY_WEEK <- as.factor(flight$DAY_WEEK)
flight$Flight.Status <- ifelse(flight$Flight.Status == 0, 'on-time','delayed')
flight$Flight.Status <- as.factor(flight$Flight.Status)

str(flight)
colnames(flight)
flight$levels <- ifelse(flight$DEP_TIME >=600 & flight$DEP_TIME <=1200, "level1",
                        ifelse(flight$DEP_TIME >=1200 & flight$DEP_TIME <=1800, "level2",
                               ifelse(flight$DEP_TIME>= 1800 & flight$DEP_TIME<=2100, "level3", "level4"))) 
flight$levels <- as.factor(flight$levels)
str(flight)

#Drop the Departure time column 
flight <- flight[ ,!colnames(flight) %in% c("DEP_TIME")]

""" Modeling """
install.packages('caret')
library(caret)
train_rows <- createDataPartition(y=flight$Flight.Status, p=0.7, list = F)
train <- flight[train_rows, ]
test <- flight[-train_rows, ]

install.packages('e1071')
library(e1071)

model_nb <- naiveBayes(train$Flight.Status ~., train)
print(model_nb)

#Measure model performance on test 
preds <- predict(model_nb, test)
confusionMatrix(data = preds, reference = test$Flight.Status)


