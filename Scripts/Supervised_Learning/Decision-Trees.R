# Decision Trees Model #
# Algorithm ====

# 1. Calculate entropy of the target field (the class label) for whole data set.
# 2. For each attribute:
#    * split the data set on the attribute
#    * calculate entropy of the target field on splitted data set, using the attribute values
#    * calculate the information gain of the attribute
# 3. Select the attribute that has the largest information gain
# 4. Branch the tree using the selected attribute
# 5. Stop, if it is a node with entropy of 0, otherwise jump to step 2.

# Libraries ====

library(data.table)
library(DataExplorer)
library(tidyverse)
library(plotly)

# Data ====

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
