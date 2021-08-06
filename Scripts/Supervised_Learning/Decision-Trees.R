# Decision Trees Model #
# Libraries ====

# Preprocessing and Visualization
library(data.table)
library(DataExplorer)
library(tidyverse)
library(plotly)

# Decision Trees Model
library(rpart) # CART methodology
library(rpart.plot)
library(tree)
library(party) # Make splits based on the conditional inference framework
library(partykit) # convert the rpart object to a party for visualization
library(caret)

# 1.0 Classification Trees Algorithm ====

# 1. Calculate entropy of the target field (the class label) for whole data set.
# 2. For each attribute:
#    * split the data set on the attribute
#    * calculate entropy of the target field on splitted data set, using the attribute values
#    * calculate the information gain of the attribute
# 3. Select the attribute that has the largest information gain
# 4. Branch the tree using the selected attribute
# 5. Stop, if it is a node with entropy of 0, otherwise jump to step 2.

# 1.1 Data ====

download.file(url = "https://ibm.box.com/shared/static/dpdh09s70abyiwxguehqvcq3dn0m7wve.data",
              destfile = "mushroom.data")

mushroom <- fread(file = "Data/mushroom.data", 
                  sep = ",", 
                  header = FALSE, 
                  stringsAsFactors = TRUE, 
                  col.names = c("Class","cap.shape","cap.surface","cap.color","bruises","odor",
                                "gill.attachment","gill.spacing", "gill.size","gill.color",
                                "stalk.shape","stalk.root","stalk.surface.above.ring",
                                "stalk.surface.below.ring","stalk.color.above.ring",
                                "stalk.color.below.ring","veil.type","veil.color",
                                "ring.number","ring.type","print","population","habitat"))

mushroom <- mushroom %>% 
  mutate(Class = factor(Class, labels = c("Edible", "Poisonous")), 
         odor = factor(odor, labels = c("Almonds","Anise","Creosote","Fishy","Foul","Musty","None","Pungent","Spicy")), 
         print = factor(print, labels = c("Black","Brown","Buff","Chocolate","Green","Orange","Purple","White","Yellow")))

# Data Exploratory Analysis 

str(mushroom)
head(mushroom)
summary(mushroom)
sapply(X = mushroom, # Missing values
       FUN = function(x) sum(is.na(x)))

ggplotly(
mushroom %>% 
  ggplot(aes(x = Class)) + 
    geom_bar(color = "black", 
             width = 0.6) + 
    #coord_flip() + 
    scale_y_continuous(breaks = seq(0, 4300, 500)) + 
    labs(title = "Frecuency by Class",
         x = "Mushroom Class", 
         y = "Frequency") + 
    theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5, 
                                    face = "bold"))
)

# DataExplorer package
mushroom[ , c("Class", "odor", "print")] %>% plot_intro(ggtheme = theme_bw()) # Variable types and missing values
mushroom[ , c("Class", "odor", "print")] %>% plot_bar(ggtheme = theme_bw()) # Bar plot
mushroom[ , c("Class", "odor", "print")] %>% plot_missing(ggtheme = theme_bw()) # Percentage missing values by variables

# 1.2 Training and Testing Data ====

set.seed(123)
training <- sample_frac(tbl = mushroom, 
                        size = 0.75, 
                        replace = FALSE)

testing <- setdiff(x = mushroom, # Rows that appear in X but not in y
                   y = training)

prop.table(table(training$Class)) # Similar proportions of target variable in both set
prop.table(table(testing$Class))

# 1.3 Classification Tree Model ====

# rpart() function 
tree_model_rpart <- rpart(Class ~ ., 
                          data = training, 
                          method = "class", # because the problem is about classification
                          parms = list(split = "gini"), # gini or information gain
                          control = rpart.control(minsplit = 20, # min number of points for a split
                                                  cp = 0.01, # Complexity parameter, tuning parameter
                                                  xval = 10, # Number of cross validations
                                                  maxdepth = 30)) # Maximum depth of any node

print(tree_model_rpart)
summary(tree_model_rpart)

tree_model_rpart$frame # Results

# Visualizing the model
rpart.plot(x = tree_model_rpart, 
           type = 3, 
           extra = 102, 
           under = TRUE, 
           faclen = 8, 
           cex = 0.75) # Text size

plot(as.party(tree_model_rpart)) # From "party" package

# 1.4 Model Performance ==== 

predictions_test <- predict(object = tree_model_rpart, 
                            newdata = testing[ , -1], 
                            type = "class")

confusionMatrix(data = testing$Class, 
                reference = predictions_test)

# 1.5 Improve the model performance - Tuning parameters ====

# Using the "caret" package and rpart function; CART approach.
names(caret::getModelInfo(model = "rpart"))

# 1.5.1 Tuning parameter: complexity parameter 
getModelInfo(model = "rpart")

set.seed(123)
tree_model_caret <- train(x = training[ , -1], # Predictors
                          y = training$Class, # Target
                          method = "rpart",
                          metric = "Accuracy",
                          control = rpart.control(minsplit = 20, # min number of points for a split
                                                  cp = 0.01, # Complexity parameter, tuning parameter, controls the tree pruning
                                                  maxdepth = 30),
                          trControl = trainControl(method = "repeatedcv", 
                                                   number = 9, #9-fold cross-validation with 2 repeat
                                                   repeats = 2), 
                          tuneGrid = data.frame(cp = seq(0.01, 0.1, 0.01)))

plot(tree_model_caret)
tree_model_caret$results # Accuracy and Kappa results
tree_model_caret$bestTune # final parameter
tree_model_caret$finalModel
tree_model_caret$times

# 1.5.2 Evaluating performance
predictions_caret <- predict(object = tree_model_caret, 
                             newdata = testing[ , -1])

confusionMatrix(testing$Class, predictions_caret)

# 1.5.3 Tuning parameter: Maximum node depth
getModelInfo(model = "rpart2")

tree_model_caret2 <- train(x = training[ , -1], # Predictors
                           y = training$Class, # Target
                           method = "rpart2",
                           metric = "Accuracy",
                           control = rpart.control(minsplit = 20, # min number of points for a split
                                                   cp = 0.01, # Complexity parameter, tuning parameter
                                                   maxdepth = 5),
                           trControl = trainControl(method = "repeatedcv", 
                                                    number = 9, #9-fold cross-validation with 2 repeat
                                                    repeats = 2), 
                           tuneGrid = data.frame(maxdepth = seq(5, 30, 5))) # In rpart function default maxdepth is 30

plot(tree_model_caret2)
tree_model_caret2$results # Accuracy and Kappa results
tree_model_caret2$bestTune # final parameter
tree_model_caret2$finalModel
tree_model_caret2$times

# 1.5.4 Evaluating performance
predictions_caret2 <- predict(object = tree_model_caret2, 
                              newdata = testing[ , -1])

confusionMatrix(testing$Class, predictions_caret2)

# 2.0 Regression Trees Algorithm ====

# From "An Introduction to Statistical Learning"

# 1. Use recursive binary splitting to grow a large tree on the training
#    data, stopping only when each terminal node has fewer than some minimum number of observations.

# 2. Apply cost complexity pruning to the large tree in order to obtain a
#    sequence of best subtrees, as a function of α.

# 3. Use K-fold cross-validation to choose α. That is, divide the training
#    observations into K folds. For each k = 1, . . . , K:

#  (a) Repeat Steps 1 and 2 on all but the kth fold of the training data.
#  (b) Evaluate the mean squared prediction error on the data in the
#      left-out kth fold, as a function of α.
#   Average the results for each value of α, and pick α to minimize the average error.

# 4. Return the subtree from Step 2 that corresponds to the chosen value of α.

# Tuning parameters: 
#   Complexity parameter
#   Maximum node depth 



