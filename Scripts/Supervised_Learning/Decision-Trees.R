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

