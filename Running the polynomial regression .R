

#Loading Required Libraries 
install.packages('tidyverse')  #pipe operator and other commands 
install.packages('caret')   #random split and cv 
install.packages('olsrr')    #ols_test_score heteroscedasticity 
install.packages('car')       #test multicolinearity vif()
install.packages('broom')     #Diagnostic metric table augment()

library(tidyverse)
library(caret)
library(olsrr)
library(car)
library(broom)

data <- read.csv('/Users/mac/Downloads/datasets-133357-317184-advertising.csv',header=TRUE)
head(data)
colSums(is.na(data))
str(data)
boxplot(data) #There are two outliers in Newspaper var 
#Drop the outliers
data <- data[-which(data$Newspaper %in% boxplot.stats(data$Newspaper)$out), ]
pairs(data,lower.panel = NULL)

#Intuition 
1. Strong positive correlation between Tv and Sales 
2. Radio and newspaper (moderate linear relationship)
3. Radio and Sales (moderate linear relationship)
4. Tv and Radio (example of no linear relationship)

#Plot corelation of interest 
plot(data$TV, data$Sales)
plot(data$Radio,data$Newspaper)
plot(data$Radio,data$Sales)
plot(data$TV,data$Radio)

#Random splitting of data into train and test set
set.seed(123)
training.samples <- data$Sales %>%
  createDataPartition(p=0.75, list = FALSE)
train_data <- data[training.samples, ]
test_data <- data[-training.samples, ]

#The simple Linear Regressions 
sm1 <- lm(Sales~TV, data = train_data)
summary(sm1)
#1. Residual error 2.29 
#2. Rsquare .81
#3. p-value 2.2e-16

sm2 <- lm(Sales ~ Radio, data = train_data)
summary(sm2)
#1. Residual error 4.91
#2. Rsquare .1355
#3. p-value 3.577e-06

sm3 <- lm(Sales ~ Newspaper, data = train_data)
summary(sm3)
#1. Residual error 5.21
#2. Rsquare .02
#3. p-value .0365 (<0.05) 

#plotting TV and Sales 
plot(train_data$TV, train_data$Sales)
#add the regression line 
abline(lm(train_data$Sales ~ train_data$TV), col= 'blue')

#Forward selection process include next variable 
mm1 <- lm(Sales ~ TV + Radio, data = train_data)
summary(mm1)

#1. Residual error 1.715
#2. Adjusted R .89
#3. p-value <2.2e-16 less than  .05 

#Test if improvement in #2 is statistically significant ANOVA

anova(sm1,mm1)
#p-value significantly less than .05 accept hypothesis 

#Diagnostic plots to check assumptions of Linear Regression
#1 The model is Linear 
plot(mm1, 1)

#2 The residuals have constant variance 
ols_test_score(mm1)

# Prob > Chi2   =    0.8962098 Greater than .05 hence homoscedastic

#3 The error terms are independent 
durbinWatsonTest(mm1)

#p-value 0.166 greater than.05 hence no autocorrelation

#Detecting multicolinearity 
vif(mm1)

#vif not greater than 5 or 10 indicates no multicolinearity

#4 The error terms have a normal distribution 
shapiro.test(mm1$residuals)

#pvalue .005 less than 0.05 hence normality does not hold

hist(mm1$residuals)

#The polynomial regression 
#Fitting second order polynomials
pm1 <- lm(Sales ~ poly(TV,2) + poly(Radio,2) + TV:Radio, data = train_data)
summary(pm1)

#adj R increased to .9258 and anova says change is statist significant

#Fitting third order polynomial to reduce chances of mltcly (orthogonal)
pm2 <- lm(Sales ~ poly(TV, 3) + poly(Radio, 3) + TV:Radio, data= train_data)
summary(pm2)
#adj R .9289 also 3rd order TV not significant hence ,Err 1.405


pm3 <- lm(Sales ~ poly(TV, 2) + poly(Radio, 3) + TV:Radio, data = train_data)
summary(pm3)
#adj R .9293 ,Residual 1.401
anova(pm1,pm3)
#pvalue .0049 very small hence statistically significant

#Check now for assumptions of LR 
#1 The model is linear
plot(pm3, 1)
#Redline at 0 linearity holds well | notice outliers

#2 Error terms have constant variance 
ols_test_score(pm3)
#Prob .3855 greater than .05 hence constant variance

#3 Error terms are independent 
durbinWatsonTest(pm3)
#pvalue .19 greater than .05 hence no correlation 

#4 Normality check 
shapiro.test(pm3$residuals)
#pvalue .05204 normality holds 
hist(pm3$residuals)

#multicolinearity check 
vif(pm3)
#all values in last column less than 5 hence no multicolinearity

#Removing the outlier 131 

#creating diagnostic metrics table 
dm =  augment(pm3)

#check minimu value of std residual 
min(dm$.std.resid)
#The above value of Studentized Residual is less than -3 (Rule of thumb), 
#Hence it indicates an outlier
which(dm$.std.resid  %in% "-3.4452042988145") #index of observ

train_data[98, ] #info 
# Removing 98th row of outlier
train_data1 = train_data %>% filter(train_data$Sales !=  1.6)

# Checking number of rows in old train data set
nrow(train_data)

# Checking number of rows in new train data set (train.data1)
nrow(train_data1)


#Now we fit the same polymodel stored in pm3
pm4 <- lm(Sales ~ poly(TV, 2) + poly(Radio, 3) + TV:Radio, data = train_data1)
summary(pm4)
#Residual 1.347, Adj R 0.9321, pvalue 2.2e-16
#we can't check anova since pm3 150obsv and pm4 149obs

#Now check assumptions of Linear Regression
#linearity
plot(pm4,1)
#constant variance homoscedasticity assumption
ols_test_score(pm4)
#Autocorrelation assumption 
durbinWatsonTest(pm4)
#NormalityTest
shapiro.test(pm4$residuals)
#multicolinearity
vif(pm4)

#Checking for outliers again by diagnostic metric table 
dm1 = augment(pm4)

# Checking minimum and maximum value of Studentized Residuals
min(dm1$.std.resid)  # -2.897961
max(dm1$.std.resid)   #2.458413
#Studentized residuals not greater than 3 hence no outliers

#Making Predictions 
prediction = pm4 %>% predict(test_data)
#check performance 
data.frame( R2 = R2(prediction, test_data$Sales),  #.95
            RMSE = RMSE(prediction, test_data$Sales), #1.14
            MAE = MAE(prediction, test_data$Sales))   #.82


#Let's do a Cross validation 
# Define training control
set.seed(123)
train_control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3)
# Train the model
model_cv <- train(Sales ~ poly(TV , 2) + poly(Radio , 3) + TV:Radio , data = data, method="lm",
                  trControl = train_control)

# Summarize the results
print(model_cv)

#RMSE      Rsquared  MAE     
#1.339603  0.937277  1.014301



























