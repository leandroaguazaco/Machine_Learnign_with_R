# K-Nearest Neighbors Algorithm # 
# Libraries ====

library(tidyverse)
library(class) # K-NN algorithm
library(caret)
library(mlbench)
library(e1071)
library(base)
library(plotly)

# Data ====

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
 
# Training and Test Data ====

set.seed(123) # Due the randomly selection

# Training data 
training <- sample_frac(tbl = Sonar, # training data: 70% from the original data set
                        size = 0.7)

# Test data
test <- setdiff(x = Sonar, y = training) # Row that appear in X but not in y

# Alternative way to select test data from the original data set
Sonar[!(rownames(Sonar) %in% rownames(training)), ] # Are the left items in the elements on the right side? 

paste("Training observations =", nrow(training), ",", "Test observations =", nrow(test))

prop.table(table(training$Class)) # Both set almost have the same proportion of Class labels
prop.table(table(test$Class))

# Training a model on data ====

# Using knn function from the 'class' library with k = 3
# Using Euclidean distance

knn_model <- knn(train = training[ , -61],  # Training set cases
                 test = test[ , -61], # Test set cases
                 cl = training[ , "Class"], # Labels of training set  
                 k = 3,
                 prob = TRUE)

