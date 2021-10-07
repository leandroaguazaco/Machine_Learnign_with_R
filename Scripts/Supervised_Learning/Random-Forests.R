# Random Forest Model #
# Libraries ====

library(tidyverse)
library(data.table)
library(DataExplorer)
library(skimr)
library(plotly)
library(randomForest)
library(caret)

# 1.0 Random Forest Algorithm ====

# From "Applied Predictive Modeling" book
# p = total amount of predictors 

# Tree based method: 
# 1 Select the number of models to build, m
# 2 for i = 1 to m do
# 3   Generate a bootstrap sample of the original data
# 4   Train a tree model on this sample
# 5   for each split do
# 6     Randomly select k (< p) of the original predictors
# 7     Select the best predictor among the k predictors and partition the data
# 8   end
# 9 Use typical tree model stopping criteria to determine when a tree is complete (but do not prune)
# 10 end

# 1.1 Tuning Parameter ====

# Random forest's tuning parameter is the number of randomly predictors 
# selected, k, to choose from at each split.

# If p = k, then this amounts simply to bagging.
# For classification k = √p and the minimum node size is one.
# For regression k = p/3 or from 2 to p and the minimum node size is five.

# k does not have drastic effect on performance.
# k tuning parameter has a serious effect on the importance predictors values.
# 1.2 Deploying Random Forests model - Classification ====

# Data: wine quality
# Target variable: quality

# 1.2.1 Data ====

# File separate by semicolon ";"

# Alternative 1: readr::read_delim() or read_csv2 functions
wine_data <- read_delim(file = "https://raw.githubusercontent.com/erood/interviewqs.com_code_snippets/master/Datasets/winequality-red.csv", 
                        delim = ";")

str(wine_data)

# Alternative 2: utils::read.csv2() function
wine_data2 <- read.csv2(file = "https://raw.githubusercontent.com/erood/interviewqs.com_code_snippets/master/Datasets/winequality-red.csv", 
                        header = TRUE, 
                        dec = ".")

wine_data2$quality <- factor(wine_data2$quality)

str(wine_data2)

# Alternative 3: data.table::fread() function
wine_data3 <- fread("https://raw.githubusercontent.com/erood/interviewqs.com_code_snippets/master/Datasets/winequality-red.csv", 
                    header = TRUE,  
                    dec = ".")
str(wine_data3)

# Exploratory Data Analysis

skim(wine_data) # Statistics Summary + histograms

wine_data %>% plot_intro(ggtheme = theme_bw()) # variables type, percentage missing values
wine_data %>% plot_histogram(ggtheme = theme_bw()) # Histograms
wine_data %>% plot_correlation(ggtheme = theme_bw()) # Correlation heat map
wine_data %>% plot_density(ggtheme = theme_bw()) # Density
wine_data %>% plot_qq(ggtheme = theme_bw()) # Testing normality by qq plots
factor(wine_data$quality) %>% plot_bar(ggtheme = theme_bw()) # Bar plot

table(factor(wine_data$quality)) # Frequency table of target variable

# 1.2.2 Training and testing data ====

set.seed(20211004) # Date 2021/10/04

wine_training <- sample_frac(tbl = wine_data2, # Training data
                             size = 0.7, 
                             replace = FALSE)

wine_testing <- setdiff(x = wine_data2, # Testing data
                        y = wine_training)

# Checking sample proportions inside samples
prop_check <- bind_cols(training = round(prop.table(table(wine_training$quality)), 2),
                        testing = round(prop.table(table(wine_testing$quality)), 2)) %>%
  mutate(wine_quality = 3:8) %>% 
  relocate(wine_quality, .before = everything())

# Class imbalances: low performance of random forests model is expected

# 1.2.3 Classification - randomForest package ====

# Without tuning parameters

set.seed(20211004) # Date 2021/10/04

rf_model <- randomForest(x = wine_training[ , 1:11], 
                         y = wine_training[ , 12],
                         ntree = 1000, # Default 500
                         # mtry = 3, # Default according to problem type: regression or classification
                         importance = TRUE)

print(rf_model)
importance(rf_model) # Variable importance
varImpPlot(rf_model, 
           main = "Varible Importance Plot") # Plot variable importance

# Assessing model performance
prediction_rf <- predict(object = rf_model, 
                         newdata = wine_testing[ , 1:11], 
                         type = "class")

caret::confusionMatrix(prediction_rf, 
                       wine_testing[ , 12])

# 1.2.4 Classification - caret package ====

# With tuning parameters: number of random selected predictors  (k or mtry)

names(getModelInfo())
getModelInfo(model = "rf") # Random forest model

set.seed(20211004) # Date 2021/10/04
rf_model_caret <- train(x = wine_training[ , 1:11], 
                        y = wine_training[ , 12], 
                        method = "rf", 
                        ntree = 1000, # Options to randomForest
                        importance = TRUE, # Options to randomForest
                        preProcess = c("center", "scale"), # Standardization
                        metric = "Kappa", # "Accuracy" or "ROC". Performance measure
                        trControl = trainControl(method = "repeatedcv", 
                                                 number = 5, #5-fold cross-validation with 2 repeats
                                                 repeats = 2,),
                                                #summaryFunction = twoClassSummary, 
                                                #classProbs = TRUE,
                                                #savePredictions = TRUE), # ROC for performance measure)
                        tuneGrid = data.frame(mtry = seq(2, 10, 1))) # There are 11 predictor variables

plot(rf_model_caret)
rf_model_caret # Best tuning parameters
rf_model_caret$results
rf_model_caret$bestTune
rf_model_caret$finalModel # This is the final decision for tuning parameters
rf_model_caret$times # Calculation time 
rf_model_caret$preProcess # if it is not taken into account, reduces the calculation time

# Assessing model performance
prediction_rf_caret <- predict(object = rf_model_caret, 
                               newdata = wine_testing[ , 1:11], 
                               type = "raw")

caret::confusionMatrix(prediction_rf_caret, 
                       wine_testing[ , 12])

# Variables Importance 
varImp(rf_model_caret)
plot(varImp(rf_model_caret))

# Conclusion: low performance model for this case study due the class imbalance inside sample
# 1.3 Deploying Random Forests Model - Regression ====

# Data: housing sales 
# Target variable: Sldprice - House sale price

# 1.3.1 Data ====

sales_data <- read.csv("https://ibm.box.com/shared/static/fzceg5vdj9hxpf7aopgvfgobi1g4vb4v.csv", 
                       dec = ".")
str(sales_data)

# Exploratory Data Analysis

sapply(sales_data, function(x) sum(is.na(x))) # Missing values amount by variable

sales_data <- sales_data %>% 
  filter(!is.na(hh_avinc)) %>% # Delete missing values
  mutate(across(.cols = -c(sldprice:d_cbd, hh_avinc), # Changing variable type
                .fns = as.factor))

skim(sales_data) # Statistics summary

sales_data %>% plot_intro(ggtheme = theme_bw()) # Variables type and missing values and rows

sales_data %>% # Quantitative variables: Histogram
  select(sldprice:d_cbd, hh_avinc) %>% 
  rename(No_of_Bedrooms = beds,
         No_of_Rooms = rooms, 
         House_Sale_Price = sldprice,
         Distance_to_Centre  = d_cbd, 
         Average_Income = hh_avinc) %>% 
  plot_histogram(geom_histogram_args = list(color = "black"), 
                 title = "Numeric variables Histogram", 
                 ggtheme = theme_bw(), 
                 theme_config = theme(plot.title = element_text(hjust = 0.5)), 
                 ncol = 3)


sales_data %>% # Quantitative variables: normality testing
  select(sldprice:d_cbd, hh_avinc) %>% 
  rename(No_of_Bedrooms = beds,
         No_of_Rooms = rooms, 
         House_Sale_Price = sldprice,
         Distance_to_Centre  = d_cbd, 
         Average_Income = hh_avinc) %>% 
  plot_qq(title = "Numeric variables QQ Plot", 
          ggtheme = theme_bw(), 
          theme_config = theme(plot.title = element_text(hjust = 0.5)), 
          ncol = 3)

sales_data %>% # Quantitative variables: correlation
  select(sldprice:d_cbd, hh_avinc) %>% 
  plot_correlation(title = "Correlation Heatmap", 
                   cor_args = list(method = "spearman"),
                   ggtheme = theme_bw(), 
                   theme_config = theme(legend.position = "right", 
                                        plot.title = element_text(hjust = 0.5), 
                                        axis.title = element_blank(), 
                                        legend.title = element_blank()))

sales_data %>% # Qualitative variables: barplot
  select(-c(sldprice:d_cbd, hh_avinc)) %>% 
  plot_bar(ggtheme = theme_bw())

# 1.3.2 Training and testing data ====

set.seed(20211006) # Date 2021/10/06
sales_training <- sample_frac(tbl = sales_data, # Training data
                              size = 0.7, 
                              replace = FALSE)

sales_testing <- setdiff(x = sales_data, # Testing data
                         y = sales_training)

# 1.3.3 Regression - randomForest package ====

set.seed(20211006) # Date 2021/10/06

rf_regression <- randomForest(x = sales_training[ , 2:11], 
                              y = sales_training[ , 1],
                              ntree = 1000, # Default 500
                              # mtry = 3, # Default according to problem type: regression or classification
                              importance = TRUE)

print(rf_regression)

# Variable Importance
importance(rf_regression)
varImpPlot(rf_regression, 
           main = "Varible Importance Plot") # Plot variable importance

# Assessing model performance
prediction_rf <- predict(object = rf_regression, 
                         newdata = sales_testing[ , 2:11])

caret::R2(pred = prediction_rf, 
          obs = sales_testing[ , 1])

caret::RMSE(pred = prediction_rf, 
            obs = sales_testing[ , 1])

# 1.3.4 Regression - caret package ====

set.seed(20211006) # Date 2021/10/06
rf_regression_caret <- train(x = sales_training[ , 2:11], 
                             y = sales_training[ , 1], 
                             method = "rf", 
                             ntree = 700, 
                             importance = TRUE, 
                             #preProcess = c("center", "scale"), 
                             metric = "Rsquared", 
                             trControl = trainControl(method = "repeatedcv", 
                                                      number = 5,#), 
                                                      repeats = 2), 
                             tuneGrid = data.frame(mtry = seq(2, 10)))

rf_regression_caret
rf_regression_caret$results
rf_regression_caret$bestTune 
rf_regression_caret$finalModel

# Variables Importance 
varImp(rf_regression_caret)
plot(varImp(rf_regression_caret))

# Assessing model performance
prediction_rf_caret <- predict(object = rf_regression_caret, 
                               newdata = sales_testing[ , 2:11])

caret::R2(pred = prediction_rf_caret, 
          obs = sales_testing[ , 1])

caret::RMSE(pred = prediction_rf_caret, 
            obs = sales_testing[ , 1])
