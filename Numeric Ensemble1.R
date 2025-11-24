player_data<-read.csv("player_data.csv",
                      stringsAsFactors = F)
#check for na's 
table(is.na(player_data))
str(player_data)
max(player_data,na.rm = T)
max(player_data)
#check the numeric
colnames(player_data)
table(is.na(player_data$TrainingHours))
table(is.na(player_data$FitnessScore))
table(is.na(player_data$Experience))

#impute median into missing values
for(col in c(3,4)){
  player_data[[col]][is.na(player_data[col])]<-median(player_data[[col]],na.rm=T)
}
#univariant distribution
library(ggplot2)
ggplot(player_data,
       aes(x=FitnessScore))+
  geom_histogram(bins = 20)+
  theme_minimal()

#check for outliers
ggplot(player_data,
       aes(x=TrainingHours))+
  geom_boxplot()

#pairwise scatter plots and correlation matrix
pairs(player_data[,c(2,3,4,5,6)])
cor(player_data[,c(2,3,4,5,6)])
library(corrplot)
corrplot::corrplot(cor(player_data[,2:6]),
                   method = "number")
#outlieer detection on performance
#perform simple z score 
z<-scale(player_data$Performance)
boxplot.stats(player_data$Performance)$out
# feature scaling
apply(player_data[,1:4],2 ,sd)

library(randomForest)
#splitting data set into train and test
set.seed(123)
library(caret)
train_number<-createDataPartition(player_data$Performance,
                                  p=0.7,list = F)
train<-player_data[train_number,]
test<-player_data[train_number,]


lm_model<-lm(Performance~., data = train)
rf_model<-randomForest(Performance~.,data = train,ntree=10,mtry=2)
varImpPlot(rf_model)
summary(lm_model)



#using support vector machines
library(e1071)
svm_model<-svm(Performance~.,data = train,
               distribution="gaussian")
summary(svm_model)

#using gradient boosting
library(gbm)
gbm_model<-gbm(Performance~., data = train,
               distribution = "gaussian",
               n.trees = 200,interaction.depth = 3,
               shrinkage = 0.05,cv.folds = 5, verbose = F)

#determining the optimal iterations
best_iter<-gbm.perf(gbm_model,method = "cv")



#performing KNN
library(FNN)
knn_pred<-knn.reg(train=train[,1:4],test=test[,1:4],
                  y=train$Performance,k=5)$pred
summary(knn_pred)

#using bagging ensemble
bag_model<-randomForest(Performance ~.,data=train,
                        ntree=200,mtry=4,replace=T)
library(randomForest)

#the prediction
?randomForest

lm_pred<-predict(lm_model,newdata = test)
rf_pred<-predict(rf_model,newdata = test)
svm_pred<-predict(svm_model,newdata=test)
gbm_pred<-predict(gbm_model,newdata = test,
                  n.trees = best_iter)
bag_pred<-predict(bag_model,newdata = test)




library(ggplot2)
residuals_df<-data.frame(
  model=rep(c("linear reg","random forest","svm","gradient boosting",
              "Knn","bagging ensemble"),
  each=nrow(test)),predicted=c(lm_pred,rf_pred,
                        svm_pred,gbm_pred,knn_pred,
                                         bag_pred),
  residual=c(test$Performance-lm_pred,
             test$Performance-rf_pred,
             test$Performance-svm_pred,
             test$Performance-gbm_pred,
             test$Performance-knn_pred,
             test$Performance-bag_pred)
)


View(residuals_df)


library(Metrics)

model_names <- c("linear reg", "random forest", "svm", "gradient boosting", "Knn", "bagging ensemble")
predictions <- list(lm_pred, rf_pred, svm_pred, gbm_pred, knn_pred, bag_pred)

results <- data.frame(
  Model = model_names,
  MAE = sapply(predictions, function(pred) mae(test$Performance, pred)),
  RMSE = sapply(predictions, function(pred) rmse(test$Performance, pred)),
  R2 = sapply(predictions, function(pred) cor(test$Performance, pred)^2)
)

print(results)

ggplot(residuals_df, aes(x = predicted, y = residual, color = model)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  facet_wrap(~model, scales = "free") +
  theme_minimal() +
  labs(title = "Residuals vs Predicted Values", x = "Predicted Performance", y = "Residuals")




actual_vs_pred <- data.frame(
  Actual = rep(test$Performance, 6),
  Predicted = unlist(predictions),
  Model = rep(model_names, each = nrow(test))
)

ggplot(actual_vs_pred, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~Model, scales = "free") +
  theme_minimal() +
  labs(title = "Actual vs Predicted Performance", x = "Actual", y = "Predicted")
