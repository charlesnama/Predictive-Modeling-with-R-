
'''
Problem Statement - Determine the price of houses in Boston area.
*Process outline* 

- Import and perform exploratory analysis
- Preprocessing of data 
- Modeling and Evaluation 

Variable explanation below. 

1) __CRIM :__ Per capita Crime rate by town

2) __ZN :__ Proportion of residential land zoned for lots over 25,000 sq.ft.

3) __INDUS :__ Proportion of non-retail business acres per town

4) __CHAS :___ Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

5) __NOX :__ nitric oxides concentration (parts per 10 million)

6) __RM :__ average number of rooms per dwelling

7) __AGE :__ proportion of owner-occupied units built prior to 1940

8) __DIS :__ weighted distances to five Boston employment centres

9) __RAD :__ index of accessibility to radial highways

10) __TAX :__ full-value property-tax rate per $10,000

11) __PTRATIO :__ pupil-teacher ratio by town

12) __B :__ 1000(Bk - 0.63)^2 where Bk is the proportion of African-Americans by town

13) __LSTAT :__ Percentage of the population in the lower economic status 

14) __MEDV  :__ Median value of owner-occupied homes in multiples of $1000
'''
rm(list = ls(all=TRUE))

housing_data <- read.csv(file = '/Users/mac/Desktop/HousingData.csv', header = TRUE, sep=',')
str(housing_data)
summary(housing_data)
head(housing_data)

'''
Missing values detection per column  
'''
colSums(is.na(housing_data))
sum(is.na(housing_data)) #70

'''
Perform exploratory analysis using visualization
'''
par(mfrow=c(2,2))
plot(housing_data$LSTAT, housing_data$MV, xlab = '% population in lower class', ylab =
       'Median house price', main = 'Housing Price vs status')
plot(housing_data$CRIM, housing_data$MV,  xlab = "Per capita crime by town", ylab = 
       "Median House Price", main = "Housing Price vs Per Capita Crime")

plot(housing_data$NOX, housing_data$MV, xlab = "Nitric Oxide Concentration in ppm", ylab =
       "Median House Price",  main = "Housing Price vs NOX concentration in ppm")

plot(housing_data$RM, housing_data$MV, xlab = "average number of rooms per dwelling", ylab = 
       "Median House Price",  main = "Housing Price vs average number of rooms per dwelling")

#Correlation between the variables 
install.packages('corrplot')
library(corrplot)
corel <- cor(housing_data, use='complete.obs')
corrplot(corel, method = 'square',type = 'upper')
'''
Intuition
- RAD and TAX are highly correlated
'''
#Convert data CHAS into required categorical format 
str(housing_data)
housing_data$CHAS <- as.factor(housing_data$CHAS)
str(housing_data)

'''
Divide the data into train and test 70/30 
'''
set.seed(123)
train_rows <- sample(x = 1:nrow(housing_data), size = 0.7*nrow(housing_data))
train_rows
train_data <- housing_data[train_rows,]
test_data <- housing_data[-train_rows,]
dim(train_data) #350
dim(test_data)  #150

'''
Columnwise check of missing values / handling 
'''
cat('Missing values in train_data \n')
colSums(is.na(train_data))

cat('Missing values in test_data \n')
colSums(is.na(test_data))

#Handling
install.packages('DMwR')
library(DMwR)
sum(is.na(train_data)) #49
train_data <- centralImputation(train_data)
sum(is.na(test_data))  #21
test_data <- centralImputation(test_data)

sum(is.na(train_data))  #0
sum(is.na(test_data))   #0

''' Standardize the data with exception of target and categorical variables'''

install.packages('dplyr')
library(dplyr)
str(train_data)

train_data_s <- select(train_data, -c('MV','CHAS'))
test_data_s <- select(test_data, -c('MV','CHAS'))

train_data_ns <- select(train_data, c('MV','CHAS'))
test_data_ns <- select(test_data, c('MV','CHAS'))

install.packages('vegan')
library(vegan)
summary(train_data)

train_data_s <- decostand(x = train_data_s, method = 'standardize', MARGIN = 2)
test_data_s <- decostand(x = test_data_s, method = 'standardize', MARGIN = 2)

train_data <- cbind(train_data_ns, train_data_s)
test_data <- cbind(test_data_ns, test_data_s)

summary(train_data)
summary(test_data)

'''Creating a basic model'''
start_time <- Sys.time()
model_basic <- lm(formula = MV~. , data=(train_data))
end_time <- Sys.time()
time_taken <- end_time - start_time
time_taken #31.453 seconds

summary(model_basic)
'''
How significant is the model with an R-squared of .75
However, we will check the assumptions of LR 
'''

par(mfrow=c(2,2))
plot(model_basic)

''' Let"s look at another approach by checking for influential points
First we check for leverage, then we use Cooks distance to eliminate them
'''
#leverage
install.packages('car')
library(car)
lev <- hat(model.matrix(model_basic))
plot(lev)

#Remove records with leverage values greater than .3 
train_data_lev<-train_data[lev>0.12, ]
dim(train_data_lev)

#Cook's distance 
cook <- cooks.distance(model_basic)
plot(cook, ylab = "Cook's distance")
train_cook <- train_data[cook<.5,]
dim(train_cook)

'''Now build model without influential points'''

model_basic2 <- lm(formula=MV~. , data = train_cook)
summary(model_basic2)
#Adj Rsquare didn't change much so we build the stepAIC model 

install.packages('MASS')
library(MASS)
model_aic <- stepAIC(model_basic2, direction = 'both', trace=F)
summary(model_aic)

par(mfrow=c(2,2))
plot(model_aic)
#Adj Rsquare still didn't change much, let's check for multicolinearity 

vif(model_basic)
vif(model_aic)
#Remember RAD and TAX? Well they're multicolinear but just for emphasis let's check
cor(housing_data$RAD, housing_data$TAX, use = 'complete.obs') #.90

model_basic3 <- lm(formula = MV ~CRIM + ZN + CHAS + NOX + RM + DIS + RAD + PT + B + LSTAT,
                   data=train_data)

summary(model_basic3)
par(mfrow=c(2,2))
plot(model_basic3)

vif(model_basic3)

'''Predict the house prices'''
preds_model <- predict(model_basic3, test_data)

'''Evaluation metrics'''
#error verification on train 

regr.eval(train_data$MV, model_basic3$fitted.values)
regr.eval(test_data$MV, preds_model)


#Future scope, apply data transformation to get better results 
#Observation error metrics not far apart in both train/test 



