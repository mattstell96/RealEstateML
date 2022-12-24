#LOAD DATA & FUNCTIONS

df_train = read.csv(YOUR PATH TO BostonHousing.csv, stringsAsFactors = T)

#CLEANING TRAIN

#1. Remove Factor Variables with less than 2 levels (otherwise, OLS error)

names(df_train) = make.names(names(df_train))

features = setdiff(colnames(df_train), c("Id", "SalePrice"))
for (f in features) {
  if (any(is.na(df_train[[f]]))) 
    if (is.character(df_train[[f]])){ 
      df_train[[f]][is.na(df_train[[f]])] <- "Others"
    }else{
      df_train[[f]][is.na(df_train[[f]])] <- -999  
    }
}

column_class <- lapply(df_train,class)
column_class <- column_class[column_class != "factor"]
factor_levels <- lapply(df_train, nlevels)
factor_levels <- factor_levels[factor_levels > 1]
df_train = df_train[,names(df_train) %in% c(names(factor_levels), names(column_class))]

df_train = as.data.frame(unclass(df_train))

#2. Variable Formatting

df_train$Id = NULL
df_train$MiscVal = NULL
df_train$LotFrontage = NULL
df_train$LotArea = as.numeric(df_train$LotArea)
df_train$MasVnrArea = as.numeric(df_train$MasVnrArea)
df_train$GarageYrBlt = as.numeric(df_train$GarageYrBlt)


#3. Post-model Cleaning (dropping variables causing issues with predictions)

df_train$Condition2 = NULL
df_train$RoofMatl = NULL
df_train$Exterior1st = NULL
df_train$Exterior2nd = NULL
df_train$ExterCond = NULL
df_train$Heating = NULL


#PARTITIONING

set.seed(2021)

train_cases = sample(nrow(df_train),round(0.7*nrow(df_train)))

train = df_train[train_cases,]

test = df_train[-train_cases,]


#A. O L S  M O D E L

ols = lm(SalePrice~.,train)
ols_step = step(ols)
summary(ols)

#PREDICTIONS

pred_ols = predict(ols_step,test)

#EVALUATE

#1. OLS performance

obs = test$SalePrice

errors_ols = obs-pred_ols

MAPE_ols = mean(abs(errors_ols/obs)) #MAPE

RMSE_ols = sqrt(mean(errors_ols^2))

#2. Benchmarking

pred_bench_avg = mean(test$SalePrice)

errors_bench = obs-pred_bench_avg

MAPE_bench = mean(abs(errors_bench/obs))

RMSE_bench = sqrt(mean(errors_bench^2))


#A.2 L A S S O  R E G R E S S I O N

library(glmnet)

# define y and xi

y = train$SalePrice

vars_selected = select.list(names(train), multiple=TRUE, title = 'select your variable names', graphics = TRUE)

xi_train = data.matrix(train[,c(vars_selected)])

# Pre-model: finding optimal lambda

lasso_preliminary = cv.glmnet(xi_train, y, alpha = 1)

lambda_best = lasso_preliminary$lambda.min
lambda_best

plot(lasso_preliminary)

#Model: the best model (aka Lasso nullifies all the "useless" variables)

lasso_model = glmnet(xi_train, y, alpha=1, lambda = lambda_best)

coef(lasso_model) #use this to get the list of variables

#PREDICT

#1 transform the test set in a data matrix

xi_test = data.matrix(test[,c(vars_selected)])

pred_lasso = predict(lasso_model, s= lambda_best, xi_test)

#EVALUATE 

#1. Check bias introduction (reduced R2)

sst = sum((obs - mean(obs))^2)
sse = sum((pred_lasso - obs)^2)

R2_lasso = 1- (sse/sst)


#2. Lasso Performance

errors_lasso = obs-pred_lasso

MAPE_lasso = mean(abs(errors_lasso/obs))

RMSE_lasso = sqrt(mean(errors_lasso^2))



#B. R E G R E S S I O N    T R E E

#Loading Libraries
library(rpart)
library(rpart.plot)

#MODEL

tree = rpart(SalePrice~.,train)
rpart.plot(tree)

#PREDICT

pred_tree = predict(tree,test)

#EVALUATE

errors_tree = obs-pred_tree

MAPE_tree = mean(abs(errors_tree/obs))

RMSE_tree = sqrt(mean(errors_tree^2))

#PRUNING

stopping = rpart.control(minsplit=1, minbucket=1, cp=0)
tree_big = rpart(SalePrice~.,data=train, control=stopping)
tree_pruned = easyPrune(tree_big) #easyPrune is a function coming from an external source

rpart.plot(tree_pruned)

#PREDICT

pred_tree_pruned = predict(tree_pruned,test)

#EVALUATE

errors_tree_pruned = obs - pred_tree_pruned

MAPE_tree_pruned = mean(abs(errors_tree_pruned/obs))

RMSE_tree_pruned = sqrt(mean(errors_tree_pruned^2))