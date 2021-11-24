# 0. Libraries ====

library(tidyverse)
library(data.table)
library(DataExplorer)
library(skimr)
library(data.table)
library(MVN)
library(faraway)
library(lmtest)
library(ggstatsplot)
library(ggcorrplot)
library(plotly)
library(caret)

# 1. Data ====

#  Freedman R data set
freedman <- read.csv(file = "C:/Users/Luz Marina/Documents/Leandro_Aguazaco/Cursos/Machine_Learnign_with_R/Data/Freedman.csv")

# 2. Exploratory Data Analysis ====

str(freedman) # Original variable types
summary(freedman)
skim(freedman)

# 2.1 Types of variables ====

freedman %>% plot_intro(ggtheme = theme_bw()) 

# 2.2 Percentage of missing values by variable ====

freedman %>% 
plot_missing(ggtheme = theme_bw(), 
             theme_config = theme(legend.position = "none"))

# 2.3 Quantitative variables ====

# 2.3.1 Variables' Distribution ====

freedman %>% # Distribution: histograms
  select_if(is.numeric) %>% 
  plot_histogram(geom_histogram_args = list(color = "black"), 
                 title = "Distribution Quantitative Variables", 
                 ggtheme = theme_bw(), 
                 ncol = 2,
                 theme_config = theme(plot.title = element_text(hjust = 0.5, 
                                                                face = "bold"), 
                                      axis.title.x = element_text(face = "plain"),
                                      axis.title.y = element_text(face = "plain")))

freedman %>% # Distribution: density
  select_if(is.numeric) %>% 
  plot_density(geom_density_args = list(color = "black"), 
               title = "Density Quantitative Variables", 
               ggtheme = theme_bw(), 
               ncol = 2,
               theme_config = theme(plot.title = element_text(hjust = 0.5, 
                                                              face = "bold"), 
                                    axis.title.x = element_text(face = "plain"),
                                    axis.title.y = element_text(face = "plain")))

freedman %>% # Distribution: scale boxplot
  scale() %>%
  boxplot()

# 2.3.2 Normality Test ====

freedman %>% # qq plot
  select_if(is.numeric) %>% 
  na.omit() %>% 
  plot_qq(ggtheme = theme_bw(), 
          geom_qq_args = list(alpha = 0.3, 
                              color = "gray"), 
          title = "Normality Test",
          ncol = 2, 
          nrow = 2, 
          theme_config = theme(plot.title = element_text(hjust = 0.5)))

# Multivariate normality test: Mardia's test 
# Mardia Goodness of Fit Test: Alfa = 0.05

# H0: Similarity to the multivariate normal distribution
# H1: No similarity to the multivariate normal distribution

mvn(data = freedman, 
    mvnTest = "mardia")

mvn(data = freedman, # Emplea estimador robusto
    mvnTest = "mardia", 
    multivariateOutlierMethod = "quan", 
    showOutliers = F)

# Conclution: reject H0

# 2.3.3 Correlation ====

# Correlation Matrix

freedman %>% 
  na.omit() %>% 
  cor(method = "spearman") # Because the variables doesn't show normal distribution

# Correlation Heatmap

freedman %>% 
  select_if(is.numeric) %>% 
  na.omit() %>% 
  plot_correlation(cor_args = list(method = "spearman"), 
			 geom_text_args = list(color = "white", 
						     label.size = 0.20),
                   title = "Correlation Heatmap",
                   ggtheme = theme_bw(), 
                   theme_config = theme(axis.title.x = element_blank(),
                                        axis.text.x = element_text(angle = 45, 
                                                                   vjust = 0.4),
                                        axis.title.y = element_blank(), 
                                        plot.title = element_text(face = "bold", 
                                                                  hjust = 0.5), 
                                        legend.position = "right", 
                                        legend.title = element_blank())) + 
  viridis::scale_fill_viridis(option = "D", 
				      alpha = 0.8)

# Correlation Matrix - Hypothesis Testing

ggcorrmat(data = freedman %>% na.omit(),
          matrix.type = "full",
  	    type = "nonparametric", # Spearman
  	    colors = viridis::viridis(n = 3),
  	    title = "Correlation Matrix - Hypothesis Testing",
  	    ggtheme = theme_bw(), 
  	    #p.adjust.method = "bonferroni"
  	    #subtitle = "sleep units: hours; weight units: kilograms")
)

# 2.3.4 Relationship: Scatterplot ====

freedman %>% # Scatterplot by crime
  select_if(is.numeric) %>%
  na.omit() %>%
  plot_scatterplot(by = "crime", 
			 geom_point_args = list(color = "black"),
			 #sampled_rows = 10000, 
			 ggtheme = theme_bw(), 
			 ncol = 2)

# 3. Multiple Linear Regression ====

# Target variable: crime

# 3.1 Linear Model ====

freedman_lm_caret <- train(x = freedman[ , 1:3], 
				   y = freedman[ , 4], 
				   method = "lm")

freedman_lm <- lm(crime ~ ., # For better visualization
			data = freedman)

names(freedman_lm)

# Model Results
summary(freedman_lm)

# The density variable is not significant in the model (p value = 0.74).
# Low R squared (0.229), the model explains only the 23% variance of the response variabl.
# F-statitic = 9.5 > 1 (p value ~ 0), there is a linear relationship between the response and at least one predictor.

# 3.2 Optimal Linear Model ====

freedman_lm_opt <- lm(crime ~ population + nonwhite, # Without the density variable
                 	    data = freedman)

# Model Results
summary(freedman_lm_opt)

# Comparing Models: Alpha = 0.05

# Bi = density variable 
# H0: Bi = 0
# H1: Bi != 0

anova(freedman_lm_opt, freedman_lm)

# Conclution: no reject Ho

# Confidence intervals  
confint(freedman_lm_opt)

# 3.3 Diagnostics ====

# 3.3.1 Checking Error Assumptions ====

# a. Constant Variance (Homoscedasticity) and Linearity ====

# Looking for dispersion, no trends or patterns in both plots.
# Ideally, the residual plot will show no discernible pattern (James et al., 2021). 
# Nonlinearity in the structural part of the model diagnostic.

par(mfrow = c(2, 2)) # Change the panel layout to 2 x 2
plot(freedman_lm_opt) # 1. Upper left and bottom left graph

# Residuals against fitted values
ggplot(data = NULL,
	 mapping = aes(x = freedman_lm_opt$fitted.values, 
			   y = freedman_lm_opt$residuals)) +
  geom_point(colour = "black", 
	       fill = "gray") + 
  geom_hline(yintercept = 0, 
		 linetype = 2) + 
  geom_smooth(se = FALSE, 
		  color = "red", 
	        linetype = 1) + 
  labs(title = "Checking Error Assumptions: Constant Variance",
	 subtitle = "Residuals vs Fitted Values",
       x = expression(paste("Fitted Value ", (hat(y)))), 
	 y = expression(paste("Residual ", (hat(epsilon))))) + 
  theme_bw() +
  #theme(plot.title = element_text(hjust = 0.5))

# Square root of absolute residuals against fitted values
ggplot(data = NULL,
	 mapping = aes(x = freedman_lm_opt$fitted.values, 
			   y = sqrt(abs(freedman_lm_opt$residuals)))) +
  geom_point(colour = "black", 
	       fill = "gray") + 
  geom_smooth(se = FALSE, 
		  color = "red", 
	        linetype = 1) + 
  labs(title = "Checking Error Assumptions: Constant Variance",
	 subtitle = "Square Root of Absolute Residuals vs Fitted Values",
       x = expression(paste("Fitted Value ", (hat(y)))), 
	 y = expression(paste("Square root of absolute residuals ", (sqrt(abs(hat(epsilon))))))) + 
  theme_bw()

Conclution: No straight-line relationship between the predictors and the response, this possibly caused a low R2

# Homoscedasticity Numerial Test

# Breusch-Pagan Test: Alfa  = 0.05

# H0: Homoscedastic Residuals 
# H1: No Homoscedastic Residuals

bptest(freedman_lm_opt)

# Conclution: no reject H0.

# b. Normality Diagnostic: Normal Errors ====

plot(freedman_lm_opt) # 2. Upper right graph

# Shapiro-Wilk Normality Test: Alfa  = 0.05

# H0: Residuals are normally distributed
# H1: Residuals are not normally distributed

shapiro.test(x = resid(freedman_lm_opt))

# c. Correlated Errors ====

# Plot of successive pairs of residuals: No patterns expected
ggplot(data = NULL, 
	 mapping = aes(x = tail(residuals(freedman_lm_opt), (length(residuals(freedman_lm_opt)))-1), 
			   y = head(residuals(freedman_lm_opt), (length(residuals(freedman_lm_opt)))-1))) +
  geom_point() + 
  geom_hline(yintercept = 0, 
		 linetype = 2) + 
  geom_vline(xintercept = 0, 
		 linetype = 2) + 
  labs(title = "Checking Error Assumptions: Correlated Errors",
	 subtitle = "Plot successive pairs of residuals",
	 x = expression(hat(epsilon)[i]), 
	 y = expression(hat(epsilon)[i+1])) +
  theme_bw()

# Durbin-Watson Autocorrelation Test: Alfa = 0.05

# H0: No Correlated Errors
# H1: Correlated Errors

dwtest(freedman_lm_opt)

# Conclution: no reject H0.

# 3.3.2 Unusual Observations ====

# a. Leverages ====

# A leverage point is extreme in the predictor space. It has the potential to influence the fit, 
# but does not necessarily do so (Faraway, 2015).
# High leverage observations tend to have a sizable impact on the estimated regression line (James et al., 2021).

c_l <- 2*(sum(hatvalues(freedman_lm_opt))/nrow(freedman)) # Critic Limit: 4 estimated parameters, 110 observations
c_l

round(hatvalues(freedman_lm_opt), 3)

which(hatvalues(freedman_lm_opt) > c_l)

# Values greater than 0.055 are leverages.
# In this case, there are eight leverage values

# Studentized residual vs Leverage statistic
ggplot(data = NULL, 
	 mapping = aes(x = hatvalues(freedman_lm_opt), 
			   y = rstudent(freedman_lm_opt))) + 
  geom_point() + 
  scale_x_continuous(breaks = seq(0, 0.5, 0.05)) + 
  geom_hline(yintercept = 0, 
		 linetype = 2) + 
  geom_vline(xintercept = c_l, 
		 linetype = 2) + 
  labs(title = "Unusual Observations Diagnostic: High Leverage Points", 
	 subtitle = "Studentized residual vs Leverage statistic", 
       x = expression(paste("Leverage statistic  ", (h[i]))), 
       y = "Studentized Residuals") + 
  theme_bw()

# b. Outliers ====

# Extremes in the vector of the response variable for which the model does not fit. 
# An outlier is a point that does not fit the current model well (Faraway, 2015). 

freedman %>% 
  ggplot(mapping = aes(x = crime)) + 
    stat_boxplot(geom = "errorbar",
                 width = 0.15) +
    geom_boxplot() + 
    labs(title = "Unusual Observations Diagnostic", 
	   subtitle = "Outliers") + 
    theme_bw() + 
    theme(axis.text.y = element_blank())

# Studentized residual vs Fitted values: 
ggplot(data = NULL, 
	 mapping = aes(x = fitted.values(freedman_lm_opt), 
	    		   y = rstudent(freedman_lm_opt))) + 
  geom_point() + 
  labs(title = "Unusual Observations Diagnostic: Outliers", 
	 subtitle = "Studentized residual vs Fitted values", 
       caption = " Expected Studentized residuals values between -3 and 3", 
       x = expression(paste("Fitted Value ", (hat(y)))), 
       y = "Studentized Residuals") + 
  theme_bw()

# Studentized Residuals: Alfa = 0.05, Bonferroni correction

stud <- rstudent(freedman_lm_opt) # t Statistic

c_l_r.1 <- qt(p = 0.05/nrow(freedman), df = nrow(freedman)-dim(freedman)[2]-1, lower.tail = T)
c_l_r.1 # Critical Region Limit

c_l_r.2 <- qt(p = 0.05/nrow(freedman), df = nrow(freedman)-dim(freedman)[2]-1, lower.tail = F)
c_l_r.2

which(stud > c_l_r.2 | stud < c_l_r.1)

# No detected outliers

# c. Influential Observations ====

# An influential point is one whose removal from the dataset would cause a large change in the fit. 
# An influential point may or may not be an outlier or leverage but it will tend to have at least one of these two properties (Faraway, 2015).

# Cook's Distances
cooks_distances <- cooks.distance(freedman_lm_opt)
which(cooks_distances > 0.5)

# If the cook's distances is greater than 0.5, the point is considerated an influential observation 
# In this case, there aren't influential observations.

par(mfrow = c(2, 2))
plot(freedman_lm_opt) # 4. bottom right graph

# 3.3.3 Checking the Structure of the Model ====
  
# Structural part of the model given by EY = Xß.

# a. Partial residual plots: Linearity expected  ====

termplot(model = freedman_lm_opt, 
         partial.resid = TRUE, 
         terms = c(1:2))

# 3.4.4  Problems with the Predictors ====

# a. Changes of Scale ====

# Changes of Scale no required

# b. Collinearity ====

freedman_predictors <- model.matrix(freedman_lm_opt)[ ,-1] # Without the intercept

# Variance inflation factor (VIF)
which(vif(freedman_predictors) > 10)

# No Collinearity detected

# 3.4.5 Assumptions According Importance ====

#(Faraway, 2015):

# 1. The systematic form of the model. Non-linearity of the response-predictor relationships.
# 2. Dependence of errors - Correlation: errors no correlated. Correlation of error terms.
# 3. Nonconstant variance - Homoscedasticity. Non-constant variance of error terms.
# 4. Normalatiy: the least important assumption. 
# 5. Outliers
# 6. High-leverage points.
# 7. Collinearity.

# 4. Predictions ====

# 4.1 Prediction of a Future Value ====

predict(freedman_lm_opt,
        new = data.frame(population = 315, nonwhite = 20.7),
	  interval = "prediction")

# 4.2 Prediction of the Mean Response ====

predict(freedman_lm_opt,
        new = data.frame(population = 315, nonwhite = 20.7),
	  interval = "confidence")





