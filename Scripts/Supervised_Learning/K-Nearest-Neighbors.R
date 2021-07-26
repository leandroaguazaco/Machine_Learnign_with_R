# K-Nearest Neighbors Model # 
# Libraries ====

library(tidyverse)
library(class) # K-NN algorithm
library(caret)
library(kknn) # Weighted knn
library(mlbench)
library(e1071)
library(psych)
library(base)
library(plotly)

# 1. Data ====

?Sonar # Information about "Sonar" data set
data("Sonar") # From mlbench library
str(Sonar)
head(Sonar)

# Data exploratory analysis (EDA)

summary(Sonar) # Statistical summary by variable

sapply(X = Sonar, # Missing values by variable
       FUN = function(x) sum(is.na(x)))

# Barplot by Class
ggplotly(
Sonar %>% 
  mutate(Class = factor(x = Class,
                        labels = c("Mine", "Rock"))) %>% 
  ggplot(mapping = aes(x = Class)) + 
    geom_bar(color = "black", 
             width = 0.6) + 
    scale_y_continuous(breaks = seq(0, 120, 10)) +
    labs(title = "Barplot of 'Class' Variable", 
         y = "Count") + 
    theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5))
)

# Histogram of V1 variable by class
ggplotly(
Sonar %>% 
  mutate(Class = case_when(Class == "M" ~ "Mine", 
                           TRUE ~ "Rock")) %>% 
  ggplot(mapping = aes(x = V1)) + 
    geom_histogram(color = "black", 
                   binwidth = 0.01) + 
    scale_y_continuous(breaks = seq(0, 36, 2)) + 
    labs(title = "Histogram of V1 variable by class ", 
         x = "Energy band N°1", 
         y = "Frequency") + 
    #scale_fill_viridis_d(option = "D") + 
    facet_wrap(~ Class, 
               scales = "free", ) + 
    theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5, 
                                    vjust = 0.20, 
                                    face = "bold"), 
          axis.title.x = element_text(vjust = -0.20))
)
 
# 2. Training and Test Data ====

set.seed(123) # Due the randomly selection

# Training data 
training <- sample_frac(tbl = Sonar, # training data: 70% from the original data set
                        size = 0.7)

# Test data
test <- setdiff(x = Sonar, y = training) # Rows that appear in X but not in y

# Alternative way to select test data from the original data set
Sonar[!(rownames(Sonar) %in% rownames(training)), ] # Are the left items in the elements on the right side? 

paste("Training observations =", nrow(training), ",", "Test observations =", nrow(test))

prop.table(table(training$Class)) # Both set almost have the same proportion of Class labels
prop.table(table(test$Class))

# 3. Training a model on data ====

# Using knn function from the 'class' library with k = 3.
# Using Euclidean distance.
# Minimum number of neighbors points = k-1 

knn_model <- knn(train = training[ , -61],  # Training set cases
                 test = test[ , -61], # Test set cases
                 cl = training[ , "Class"], # Labels of training set  
                 k = 3)
knn_model

# 4. Evaluate the model performance ==== 

# Confusion matrix to evaluate the model performance
confusion_mat <- xtabs(~ test$Class + knn_model) # labels from test set vs output labels from the model
confusion_mat
table(test$Class, knn_model)

# Model performance: accuracy 
accuracy <- sum(diag(confusion_mat)) / sum(confusion_mat) # Problematic due the unbalanced data 
accuracy

# Model performance: Kappa statistic
cohen.kappa(x = data.frame(test$Class, knn_model))

# Assess k = 3
# Leave-one-out cross-validation for the training data using knn.cv function
knn_model_Loocv <- knn.cv(train = training[ , -61], 
                          cl = training$Class, 
                          k = 3)

confusion_mat_Loocv <- xtabs(~ training$Class + knn_model_Loocv) # labels from training set vs output labels from the model Loocv
confusion_mat_Loocv

accuracy_Loocv <- sum(diag(confusion_mat_Loocv)) / sum(confusion_mat_Loocv)
accuracy_Loocv

# Depending the k value, k-NN algorithm can be prone to overfitting.

# 5. Improve the model performance ====

# Choosing a suitable𝑘through repeated cross-validations using 'caret' library.
# Repeated k-Fold-Cross-Validation, k = [5, 10]

# 5.1 Training and test data
training_index <- createDataPartition(y = Sonar$Class, # Selecting indexes
                                      p = 0.7, 
                                      list = FALSE)

new_training <- Sonar[training_index, ]
new_test <- Sonar[-training_index, ]

prop.table(table(new_training$Class)) # It preserver the distribution of the outcome or target 
prop.table(table(new_test$Class))

# 5.2 function setup to do 5-fold cross-validation with 2 repeat.Parameters for train function
control <- trainControl(method = "repeatedcv", # Repeated k-fold cross validation 
                        number = 5, # 5 groups
                        repeats = 2) # 2 repetitions

# k values selected, data frame with possible tuning values
k_grid <- data.frame(k = c(1, 3, 5, 7, 9, 11))
k_grid

# 5.3 Built model through 'caret' package
names(getModelInfo()) # Classification or regression models in 'caret' package

best_model <- train(Class ~ ., # formula
                    data = new_training, 
                    method = "knn", 
                    trControl = control, 
                    preProcess = "pca", # Highly recommended in K-NN algorithm
                    tuneGrid = k_grid)

plot(best_model) # Visually assess the changes in accuracy for different choices of k

# Model Tuning 
as.data.frame(best_model$results) %>% 
  select(k, Accuracy, Kappa) %>% 
  pivot_longer(cols = -k,
               names_to = "Type", 
               values_to = "Values") %>% 
  ggplot(aes(x = k, 
             y = Values, 
             color = Type)) + 
    geom_line() + 
    geom_point() + 
    geom_vline(xintercept = 5, 
               color = "gray") + 
    scale_x_continuous(breaks = 1:11) + 
    scale_color_discrete(name = "Statistic:") + 
    scale_y_continuous(breaks = seq(0.35, 0.9, 0.05)) + 
    labs(title = "Model Tuning - KNN", 
         x = "k = Neighbors", 
         y = "Value") + 
    theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5), 
          legend.title = element_text(hjust = 0.5), 
          legend.position = "bottom",
          legend.box.background = element_rect(color = "black", 
                                               size = 0.8))

# 5.4 Evaluate model

# Performance on training data
predictions_training <- knn.cv(train = new_training[ , -61], 
                               cl = new_training$Class, 
                               k = 5)

confusionMatrix(predictions_training, new_training$Class)
cohen.kappa(data.frame(predictions_training, new_training$Class))

# Performance on test data
new_test_pred <- new_test[ , -61]

predictions_test <- predict(object = best_model, 
                            newdata = new_test_pred,
                            k = 5)

confusionMatrix(predictions_test, new_test$Class)


# 6. Weighted k-Nearest Neighbor Classifier ====

# Tuning model 
# k = neighbors, kernel, distance
wknn_training_model_LOOCV <- train.kknn(Class ~ ., # LOOCV
                                        data = new_training, 
                                        kmax = 10, 
                                        distance = 2, # Euclidean Distance
                                        kernel = c("triangular", "epanechnikov", "biweight", 
                                                   "triweight", "cos", "inv", "gaussian", "optimal"),
                                        scale = TRUE)
plot(wknn_training_model_LOOCV)

# Best parameters
wknn_training_model_LOOCV$best.parameters

# Model Performance
wknn_model <- kknn(Class ~ ., 
                   train = new_training, 
                   test = new_test, 
                   k = 4,
                   distance = 2, # q parameter in Minkowski formula = Euclidean distance if q = 2
                   kernel = "triangular")

summary(wknn_model)

confusionMatrix(wknn_model$fitted.values, new_test$Class)
