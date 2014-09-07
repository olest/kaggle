# some exploratory analysis and simple model fits for the
# data from the kaggle bike sharing demand challenge
# https://www.kaggle.com/c/bike-sharing-demand/

library(ggplot2)
library(lubridate)

train <- read.csv("train.csv", stringsAsFactors=FALSE)

# reformatting factors, taken from 
# http://beyondvalence.blogspot.co.uk/2014/06/predicting-capital-bikeshare-demand-in.html
train$season <- factor(train$season, c(1,2,3,4), ordered=FALSE)
train$holiday <- factor(train$holiday, c(0,1), ordered=FALSE)
train$workingday <- factor(train$workingday, c(0,1), ordered=FALSE)
train$weather <- factor(train$weather, c(4,3,2,1), ordered=TRUE)
#train$datetime <- as.POSIXct(train$datetime, format="%Y-%m-%d %H:%M:%S")

train$hour <-hour(train$datetime)
train$dow <- wday(train$datetime)
train$month <- month(train$datetime)
train$year <- year(train$datetime)


# inspect data
#head(train)
#str(train)

test <- read.csv("test.csv", stringsAsFactors=FALSE)
test$season <- factor(test$season, c(1,2,3,4), ordered=FALSE)
test$holiday <- factor(test$holiday, c(0,1), ordered=FALSE)
test$workingday <- factor(test$workingday, c(0,1), ordered=FALSE)
test$weather <- factor(test$weather, c(4,3,2,1), ordered=TRUE)
#test$datetime <- as.POSIXct(test$datetime, format="%Y-%m-%d %H:%M:%S")

test$hour <- hour(test$datetime)
test$dow <- wday(test$datetime)
test$month <- month(test$datetime)
test$year <- year(test$datetime)

train.lm <- lm(count ~ ., data=train[,-c(1,10,11)])
#summary(train.lm)

test.pred.lm <- predict(train.lm,test[,-1])
test.pred.lm[test.pred.lm<0] <- 0
# create output file
output <- data.frame(datetime=test$datetime, count=as.integer(test.pred.lm))
write.csv(output, file="results.lm.csv", quote=FALSE, row.names=FALSE)


