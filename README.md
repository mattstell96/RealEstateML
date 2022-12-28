<h1> Machine Learning Applications To Housing Real Estate: Selling Price Prediction Algorithm </h1>

<p align="center">

<h2>Abstract</h2>

The objective of this project is to build an algorithm that accurately predicts the selling price of houses. The work is based on a sample of houses sold in Ames, Iowa. Firstly, pre-processing steps are carried out on the original dataset to ensure data quality and technical feasibility. Secondly, having designed, tested, and optimized four predictive models, namely OLS Regression, Lasso Regression, Regression Tree, and Random Forest, we demonstrate the Random Forest algorithm to be the most accurate algorithm. <br />

💡 This project was carried out for my Machine Learning class with N. Karst at Babson College<br/>

<br />

<h2>R Libraries and Utilities</h2>

 - <b>rpart</b>
 - <b>glmnet</b>

*NOTE: the pruning of the classification tree was carried out with an external function*
 
<br />

<h2>Environments Used </h2>

- <b>macOS Monterey</b>

<br />

<h2>Project walk-through:</h2>

<br />
 
<h3> Preliminary </h3>

**Step 1. Load & Clean Data** <br/>

```r
#Load Data
df_train = read.csv(YOUR PATH TO BostonHousing.csv, stringsAsFactors = T)

#Remove Factor Variables with less than 2 levels (otherwise, OLS error)
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

#Variable formatting
df_train$Id = NULL
df_train$MiscVal = NULL
df_train$LotFrontage = NULL
df_train$LotArea = as.numeric(df_train$LotArea)
df_train$MasVnrArea = as.numeric(df_train$MasVnrArea)
df_train$GarageYrBlt = as.numeric(df_train$GarageYrBlt)
```

In particular, based on previous iterations, the following variables are removed to prevent errors with the algorithms.

```r
df_train$Condition2 = NULL
df_train$RoofMatl = NULL
df_train$Exterior1st = NULL
df_train$Exterior2nd = NULL
df_train$ExterCond = NULL
df_train$Heating = NULL
```

<br />

**Step 2. Partitioning** <br/>

```r
set.seed(2021)

train_cases = sample(nrow(df_train),round(0.7*nrow(df_train)))

train = df_train[train_cases,]

test = df_train[-train_cases,]

```

<br />

<h3> Modeling </h3>

**Step 3. Ordinary Least Squares (OLS)** <br/>

```r
#Build model
ols = lm(SalePrice~.,train)
ols_step = step(ols)
summary(ols)

#Predict
pred_ols = predict(ols_step,test)

#Evaluate predictive performance
obs = test$SalePrice

errors_ols = obs-pred_ols

MAPE_ols = mean(abs(errors_ols/obs)) #MAPE

RMSE_ols = sqrt(mean(errors_ols^2))

#Performance benchmarking
pred_bench_avg = mean(test$SalePrice)

errors_bench = obs-pred_bench_avg

MAPE_bench = mean(abs(errors_bench/obs))

RMSE_bench = sqrt(mean(errors_bench^2))
```

Visualization of OLS Regression: Relationship between Predictions and Observations
<img src="https://i.imgur.com/Vqq0Xd5.png" height="80%" width="80%" alt="OLS Chart"/> <br/>

<br />

**Step 5. Lasso Regression** <br/>

```r
#Load library
library(glmnet)

#Define y and xi
y = train$SalePrice

vars_selected = select.list(names(train), multiple=TRUE, title = 'select your variable names', graphics = TRUE)

xi_train = data.matrix(train[,c(vars_selected)])

# Pre-model: finding optimal lambda
lasso_preliminary = cv.glmnet(xi_train, y, alpha = 1)

lambda_best = lasso_preliminary$lambda.min
lambda_best

plot(lasso_preliminary)

#Build model
lasso_model = glmnet(xi_train, y, alpha=1, lambda = lambda_best)

coef(lasso_model) #use this to get the list of variables

#Predict
xi_test = data.matrix(test[,c(vars_selected)])

pred_lasso = predict(lasso_model, s= lambda_best, xi_test)

#Predictive performance evaluation
sst = sum((obs - mean(obs))^2)
sse = sum((pred_lasso - obs)^2)

R2_lasso = 1- (sse/sst)

#Performance benchmarking
errors_lasso = obs-pred_lasso

MAPE_lasso = mean(abs(errors_lasso/obs))

RMSE_lasso = sqrt(mean(errors_lasso^2))
```

Visualization of Lasso Regression: Relationship between Predictions and Observations<br/>
<img src="https://i.imgur.com/BoQecDM.png" height="80%" width="80%" alt="Lasso Chart"/> <br/>

<br />

**Step 6. Regression Tree** <br/>

```r
#Load libraries
library(rpart)
library(rpart.plot)

#Build model

tree = rpart(SalePrice~.,train)
rpart.plot(tree)

#Predict

pred_tree = predict(tree,test)

#Predictive performance evaluation

errors_tree = obs-pred_tree

MAPE_tree = mean(abs(errors_tree/obs))

RMSE_tree = sqrt(mean(errors_tree^2))

#Tree pruning

stopping = rpart.control(minsplit=1, minbucket=1, cp=0)
tree_big = rpart(SalePrice~.,data=train, control=stopping)
tree_pruned = easyPrune(tree_big) #easyPrune is a function coming from an external source

rpart.plot(tree_pruned)

#Predict

pred_tree_pruned = predict(tree_pruned,test)

#Predictive performance evaluation

errors_tree_pruned = obs - pred_tree_pruned

MAPE_tree_pruned = mean(abs(errors_tree_pruned/obs))

RMSE_tree_pruned = sqrt(mean(errors_tree_pruned^2))
```

Regression Tree Visualization: <br/>
<img src="https://i.imgur.com/pSELNw7.png" height="80%" width="80%" alt="Regression Tree"/> <br/>

Random Forest Visualization: Predictors Importance <br/>
<img src="https://i.imgur.com/ydFXlS7.png" height="80%" width="80%" alt="Regression Tree"/> <br/>

<br />

<h3> Meta-Analysis: Models Comparison </h3>

<img src="https://i.imgur.com/VMFQI4F.png" height="80%" width="80%" alt="Regression Tree"/> <br/>

**Conclusion**
It can be pointed out that the Random Forest and the OLS Regression are respectively more accurate than the Regression Tree and the Lasso Regression. Random Forest is also the most accurate model of all, although there is only a small performance difference between this model and the OLS Regression. 
The reader must once again be reminded that in machine learning applications there is a tradeo� between accuracy (low bias) and flexibility (low variance), meaning that the more accurate and complex models are, the more likely to generate estimation errors when they are fed with new and unseen data. Therefore, it is important to stress the fact that in this context models are compared in terms of accuracy only.


</p>



<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
