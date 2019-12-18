rm(list = ls())
setwd("C:/Users/michael/Desktop/edWisor/PROJECT/cab fare prediction")
getwd()

library(outliers)
library(ggplot2)
library(ggExtra)
library(gridExtra)
library(corrgram)
library(rpart)
library(MASS)
library(DMwR)
library(caret)
library(plyr)
library(randomForest)
library(usdm)
library(DataCombine)
library(naniar)
library(pracma)
library(schoolmath)
library(corrgram)
library(corrplot)

#loading datasets
train = read.csv('train_cab.csv', header = T)
test = read.csv('test.csv', header = T)

str(train)
str(test)
dim(train)
dim(test)
head(train, 5)
head(test, 5)
summary(train$passenger_count)
summary(train$fare_amount)

train$fare_amount = as.numeric(as.character(train$fare_amount))
str(train$fare_amount)
str(train$passenger_count)
train = train[train$fare_amount >= 0, ]
train = train[which(train$passenger_count <= 6), ]
train = train[which(train$passenger_count > 0), ]
str(train)


#Converting timestamp into new features
train$pickup_date = as.Date(as.character(train$pickup_datetime))
train$pickup_weekday = as.factor(format(train$pickup_date,"%u"))# Monday = 1
train$pickup_mnth = as.factor(format(train$pickup_date,"%m"))
train$pickup_yr = as.factor(format(train$pickup_date,"%Y"))
pickup_time = strptime(train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
train$pickup_hour = as.factor(format(pickup_time,"%H"))

#Converting datatype to numeric
train$pickup_weekday = as.numeric(as.character(train$pickup_weekday))
str(train$pickup_weekday)
train$pickup_mnth = as.numeric(as.character(train$pickup_mnth))
train$pickup_yr = as.numeric(as.character(train$pickup_yr))
train$pickup_hour = as.numeric(as.character(train$pickup_hour))
str(train)
train$pickup_date = as.Date(as.POSIXct(train$pickup_date, format="%Y-%m-%d %H:%M:%S"))
str(train$pickup_date)


which(is.na(train$fare_amount))
which(is.na(train$dist))
which(is.na(train$pickup_date))
train = na.omit(train)
dim(train)

print(paste('pickup_longitude above 180=',nrow(train[which(train$pickup_longitude >180 ),])))
print(paste('pickup_longitude above -180=',nrow(train[which(train$pickup_longitude < -180 ),])))
print(paste('pickup_latitude above 90=',nrow(train[which(train$pickup_latitude > 90 ),])))
print(paste('pickup_latitude above -90=',nrow(train[which(train$pickup_latitude < -90 ),])))
print(paste('dropoff_longitude above 180=',nrow(train[which(train$dropoff_longitude > 180 ),])))
print(paste('dropoff_longitude above -180=',nrow(train[which(train$dropoff_longitude < -180 ),])))
print(paste('dropoff_latitude above -90=',nrow(train[which(train$dropoff_latitude < -90 ),])))
print(paste('dropoff_latitude above 90=',nrow(train[which(train$dropoff_latitude > 90 ),])))

nrow(train[which(train$pickup_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
nrow(train[which(train$dropoff_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
# there are values which are equal to 0. we will remove them.
train = train[-which(train$pickup_latitude > 90),]
train = train[-which(train$pickup_longitude == 0),]
train = train[-which(train$dropoff_longitude == 0),]
train = train[-which(train$passenger_count == 0.12),]
train = train[-which(train$passenger_count == 1.3),]
train = train[-which(train$fare_amount == 0),]


#Converting and calculating distance
#haversine function
deg_to_rad = function(deg){
  (deg * pi) / 180
}
haversine = function(long1,lat1,long2,lat2){
  #long1rad = deg_to_rad(long1)
  phi1 = deg_to_rad(lat1)
  #long2rad = deg_to_rad(long2)
  phi2 = deg_to_rad(lat2)
  delphi = deg_to_rad(lat2 - lat1)
  dellamda = deg_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters
}


train$dist = haversine(train$pickup_longitude, train$pickup_latitude, train$dropoff_longitude, train$dropoff_latitude)
train = train[-which(train$dist == 0),]



#################################### Exploratary Data Analysis ################################################
#MISSING VALUE ANALYSIS
missing_values = data.frame(apply(train, 2, function(x){sum(is.na(x))}))
gg_miss_var(train) + labs(y = "Missing values")

missing_values = data.frame(apply(test, 2, function(x){sum(is.na(x))}))
gg_miss_var(test) + labs(y = "Missing values")


#outlier analysis
#boxplot merthod

plot1 = ggplot(train,aes(x = fare_amount,y = dist))
plot1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

#replacing with NA
vals1 = train[,"dist"] %in% boxplot.stats(train[,"dist"])$out
train[which(vals1),"dist"] = NA

#imputing the NA 
train = knnImputation(train, k=3)


plot2 = ggplot(train,aes(x = factor(passenger_count),y = fare_amount))
plot2 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

#replacing with NA 
vals2 = train[,"fare_amount"] %in% boxplot.stats(train[,"fare_amount"])$out
train[which(vals2),"fare_amount"] = NA

sum(is.na(train$fare_amount))

#Imputing with KNN
train = knnImputation(train,k=3)

sum(is.na(train$fare_amount))
str(train)

summary(train)
#CORRELATION ANALYSIS 
cor(train)
corplot = cor(train)
corrplot(corplot)

a1 = aov(fare_amount ~ passenger_count + pickup_hour + pickup_yr + pickup_mnth + pickup_weekday, data = train)
summary(a1)

#removing unwanted variables
train = subset(train, select = -c(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, pickup_date))
rmExcept(keepers = "train","test")

#Multicollinearity
vif(train[,-1])
vifcor(train[,-1], th = 0.9)

#VISUALIZATIONS
bar_graph1 = ggplot(data = train, aes(x = passenger_count)) + geom_bar() + ggtitle("Count of passengers")
gridExtra::grid.arrange(bar_graph1)

bar_graph3 = ggplot(data = train, aes(x = pickup_weekday)) + geom_bar() + ggtitle("Weekdays")
gridExtra::grid.arrange(bar_graph3)

bar_graph4 = ggplot(data = train, aes(x = pickup_mnth)) + geom_bar() + ggtitle("Months")
gridExtra::grid.arrange(bar_graph4)

bar_graph5 = ggplot(data = train, aes(x = pickup_yr)) + geom_bar() + ggtitle("Year")
gridExtra::grid.arrange(bar_graph5)

bar_graph6 = ggplot(data = train, aes(x = pickup_hour)) + geom_bar() + ggtitle("Hours")
gridExtra::grid.arrange(bar_graph6)


truehist(train$fare_amount)
lines(density(train$fare_amount))
qqnorm(train$fare_amount)
qqnorm(train$dist)

is.negative(train$fare_amount)
barplot(train$fare_amount, 
        horiz=TRUE, 
        xlim=c(-500,500), 
        xlab="fare amount", 
        axisnames=FALSE)
is.na(train$fare_amount)

#NORMALIZATION
train[,'dist'] = (train[,'dist'] - min(train[,'dist']))/
  (max(train[,'dist'] - min(train[,'dist'])))
qqnorm(train$dist)

barplot(train$dist,
        horiz=TRUE,
        xlab="Distance", 
        axisnames=FALSE)

is.negative(train$dist)

########################################### MODEL DEVELOPMENT ################################################
#Dividing the data into train and test
set.seed(1234)
data_split=sample(1:nrow(train), nrow(train)*0.8)
train_data=train[data_split,]
test_data=train[-data_split,]
dim(train_data)
dim(test_data)


######################## LINEAR REGRESSION #####################
#MAPE : 19.71
#RMSE : 2.45
#MSE : 6.00
#MAE : 1.67
#R-squared : 64.98
#Adjusted R-squared : 64.97

model1 = lm(fare_amount~., data = train_data)
summary(model1)
model1_pred = predict(model1, test_data[,2:7])

#mape
mape=function(actual, predicted) {
  mean(abs((actual - predicted)/actual))*100
}
mape(test_data[,1], model1_pred)
regr.eval(trues = test_data[,1], pred = model1_pred, stats = c("mae", "mse", "rmse", "mape"))


######################### DECISION TREE ########################
#MAPE : 22.91
#RMSE : 2.63
#MSE : 6.95
#MAE : 1.89

model2 = rpart(fare_amount ~., data = train_data, method = "anova")
model2_pred = predict(model2, test_data[,-1])
mape(test_data[,1], model2_pred)
regr.eval(trues = test_data[,1], pred = model2_pred, stats = c("mae", "mse", "rmse", "mape"))
str(test_data)
str(train_data)

########################### RANDOM FOREST ########################
#MAPE : 18.92
#RMSE : 2.36
#MSE : 5.57
#MAE : 1.58

model3 = randomForest(fare_amount~., data = train_data, ntree = 500)
pred_model3 = predict(model3, test_data[,-1])
mape(test_data[,1], pred_model3)
regr.eval(trues = test_data[,1], pred = pred_model3, stats = c("mae", "mse", "rmse", "mape"))


write.csv(pred_model3,"final_cab_fare_predictions_R.csv", row.names = FALSE)

