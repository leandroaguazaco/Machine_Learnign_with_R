# K-Nearest Neighbors Algorithm # 
# Libraries ====

library(tidyverse)
library(class)
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
 
