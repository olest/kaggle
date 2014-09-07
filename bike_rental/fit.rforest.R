# fitting a random forest to the data from the kaggle bike sharing competition
# https://www.kaggle.com/c/bike-sharing-demand/

library(randomForest)
#library(ggplot2)
library(lubridate)

train <- read.csv("train.csv", stringsAsFactors=FALSE)
test <- read.csv("test.csv", stringsAsFactors=FALSE)

# reformatting factors, taken from 
# http://beyondvalence.blogspot.co.uk/2014/06/predicting-capital-bikeshare-demand-in.html
train$season <- factor(train$season, c(1,2,3,4), ordered=FALSE)
train$holiday <- factor(train$holiday, c(0,1), ordered=FALSE)
train$workingday <- factor(train$workingday, c(0,1), ordered=FALSE)
train$weather <- factor(train$weather, c(4,3,2,1), ordered=TRUE)

# not sure if it's better to encode these as categorial variables
train$hour <-hour(train$datetime)
train$dow <- wday(train$datetime)
train$month <- month(train$datetime)
train$year <- year(train$datetime)

# inspect data
#head(train)
#str(train)

test$season <- factor(test$season, c(1,2,3,4), ordered=FALSE)
test$holiday <- factor(test$holiday, c(0,1), ordered=FALSE)
test$workingday <- factor(test$workingday, c(0,1), ordered=FALSE)
test$weather <- factor(test$weather, c(4,3,2,1), ordered=TRUE)

test$hour <- hour(test$datetime)
test$dow <- wday(test$datetime)
test$month <- month(test$datetime)
test$year <- year(test$datetime)

train.rforest <- randomForest(count ~ ., data=train[,-c(1,10,11)],ntree=700)
#summary(train.rforest)

test.pred.rf <- predict(train.rforest,test[,-1])
#test.pred.rf[test.pred.rf<0] <- 0

# create output file
output <- data.frame(datetime=test$datetime, count=as.integer(test.pred.rf))
write.csv(output, file="results.rforest.n700.csv", quote=FALSE, row.names=FALSE)


