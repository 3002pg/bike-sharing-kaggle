library(data.table)
library(lubridate)
library(ggplot2)
library(caret)
library(doParallel)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(rpart)
library(randomForest)

train <- fread("bike_train.csv")
test <- fread("bike_test.csv")

#extract hour, month, year and day from datetime
train$hour <- hour(ymd_hms(train$datetime))
train$month <- month(ymd_hms(train$datetime))
train$year <- year(ymd_hms(train$datetime))
train$day <- wday(ymd_hms(train$datetime))


test$hour <- hour(ymd_hms(test$datetime))
test$month <- month(ymd_hms(test$datetime))
test$year <- year(ymd_hms(test$datetime))
test$day <- wday(ymd_hms(test$datetime))


#segregate numerical and categorical variables for visualization
numCols <- c("temp", "humidity", "windspeed")
catCols <- c("season", "year", "month", "hour", "holiday", "workingday", "weather", "day")

#convert season, holiday, workingday and weather to factors
train[,(catCols):= lapply(.SD, factor), .SDcols = catCols]
test[,(catCols) := lapply(.SD, factor), .SDcols = catCols]

train <- as.data.frame(train)
test <- as.data.frame(test)


#Visualizations
bike.scatter <- function(df, cols){
  require(ggplot2)
  for(col in cols){
    p1 <- ggplot(df, aes_string(x = col, y = "count")) +
      geom_point(aes(alpha = 0.001, color = "blue")) +
      geom_smooth(method = "loess") +
      ggtitle(paste("Count of bikes rented vs.", col)) +
      theme(text = element_text(size=16))
    print(p1)
  }
}

bike.box <- function(df, cols){
  require(ggplot2)
  for(col in cols){
    p1 <- ggplot(df, aes_string(x = col, y = "count", group = col)) +
      geom_boxplot() +
      ggtitle(paste("Counts of bike rented vs", col)) +
      theme(text = element_text(size=16))
    print(p1)
  }
}

bike.hist <- function(df, cols){
  require(ggplot2)
  for(col in cols){
    p1 <- ggplot(df, aes_string(x = col)) +
      geom_histogram() +
      ggtitle(paste('Density of', col)) +
      theme(text = element_text(size=16))
    print(p1)
  }
}

bike.scatter(train, numCols)
bike.box(train, catCols)
bike.hist(train, numCols)

## FEATURE ENGINEERING
#convert hour and month into an integer
train$hour <- as.integer(train$hour)
test$hour <- as.integer(test$hour)
train$month   <- as.integer (train$month)

test$registered = 0
test$casual = 0
test$count = 0
data = rbind(train, test)


#create different hour buckets for casual users
fancyRpartPlot(rpart(casual~hour, data = train))

data$dp_cas <- 0
data$dp_cas[data$hour <= 8] = 1
data$dp_cas[data$hour == 9] = 2
data$dp_cas[data$hour >= 10 & data$hour <= 19] = 3
data$dp_cas[data$hour > 19] = 4

#create temperature bins
fancyRpartPlot(rpart(count~temp, data = train))

data$temp_bins <- 0
data$temp_bins[data$temp < 15] = 1
data$temp_bins[data$temp >= 15 & data$temp < 23] = 2
data$temp_bins[data$temp >= 23 & data$temp < 30] = 3
data$temp_bins[data$temp >=30] = 4

#create different day types
data$day_type <- ""
data$day_type[data$holiday == 0 & data$workingday == 0] <- "weekend"
data$day_type[data$holiday ==1] <- "holiday"
data$day_type[data$holiday == 0 & data$workingday == 1] <- "working day"

#create weekend factor
data$weekend <- 0
data$weekend[data$day == "Sun" | data$day == "Sat"] <- 1

#bin year variable to indicate growing business with every quarter
data$year_part[data$year=='2011']=1
data$year_part[data$year=='2011' & data$month>3]=2
data$year_part[data$year=='2011' & data$month>6]=3
data$year_part[data$year=='2011' & data$month>9]=4
data$year_part[data$year=='2012']=5
data$year_part[data$year=='2012' & data$month>3]=6
data$year_part[data$year=='2012' & data$month>6]=7
data$year_part[data$year=='2012' & data$month>9]=8


#divide data into train and test (1-20 for train, 20-30 for test)
train <- data[as.integer(substr(data$datetime, 9,10))<20,]
test <- data[as.integer(substr(data$datetime, 9,10))>19,]


#Model building - GBM
#set up parallel processing
registerDoParallel(4)
getDoParWorkers()

set.seed(123)
ctrl <- trainControl(method = "cv", number = 6, summaryFunction = defaultSummary, allowParallel = TRUE)
gbm.grid <- expand.grid(n.trees = 250, 
                        interaction.depth = c(30), 
                        shrinkage = c(0.1), 
                        n.minobsinnode = c(10))


fit.gbm.reg <- train(log1p(registered)~hour+day+humidity+temp_bins+windspeed+weather+year_part+month+day_type+workingday+season+atemp,
                 data = train,
                 method = "gbm",
                 trControl = ctrl,
                 tuneGrid = gbm.grid,
                 metric = "RMSE",
                 maximize = FALSE)

plot(varImp(fit.gbm.reg))

fit.gbm.cas <- train(log1p(casual)~hour+day+humidity+atemp+windspeed+weather+dp_cas+year_part+month+temp_bins+day_type+workingday+season,
                  data = train,
                  method = "gbm",
                  trControl = ctrl,
                  tuneGrid = gbm.grid,
                  metric = "RMSE",
                  maximize = FALSE)

plot(varImp(fit.gbm.cas))

pred.gbm.reg <- expm1(predict(fit.gbm.reg, test))
pred.gbm.cas <- expm1(predict(fit.gbm.cas, test))
pred.gbm <- pred.gbm.reg + pred.gbm.cas

submit <- data.frame(datetime = test$datetime, count = pred.gbm)
write.csv(submit, "bike_gbm.csv", row.names = F)

## Kaggle score - 0.38105
