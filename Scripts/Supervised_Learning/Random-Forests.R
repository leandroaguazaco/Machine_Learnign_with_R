# Random Forest Model #
# Libraries ====

library(randomForest)
library(caret)

# 1.0 Random Forest Algorithm ====

# From "Applied Predictive Modeling" book
# p = total amount of predictors 

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
# For classification k = √p.
# For regression k = p/3 or from 2 to p.

# k does not have drastic effect on performance.
# k tuning parameter has a serious effect on the importance predictors values.