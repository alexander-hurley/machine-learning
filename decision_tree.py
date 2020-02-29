# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:06:49 2019

@author: Alexander Hurley

ML 2 - Decision Tree

Dataset divorce.csv was taken from https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set
and modified by Alexander Hurley for use in the INFR 3700U final project. 

"""

# Importing Required Libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Reading data from file to the "divorce" variable
divorce = pd.read_csv("divorce.csv",
                      skiprows = 1,
                      names=["atr1", "atr2", "atr3", "atr4", "atr5", "atr6", "atr7", "atr8", "atr9", 
                             "atr10", "atr11", "atr12", "atr13", "atr14", "atr15", "atr16", "atr17", 
                             "atr18", "atr19", "atr20", "atr21", "atr22", "atr23", "atr24", "atr25", 
                             "atr26", "atr27", "atr28", "atr29", "atr30", "atr31", "atr32", "atr33", 
                             "atr34", "atr35", "atr36", "atr37", "atr38", "atr39", "atr40", "atr41", 
                             "atr42", "atr43", "atr44", "atr45", "atr46", "atr47", "atr48", "atr49", 
                             "atr50", "atr51", "atr52", "atr53", "atr54", "divorce"])

# The variable to test in the decision tree
atr = ["atr31"]     # atr31 is the level of aggression one feels against their spouse.
                    # This tree will show potential outcomes, leading to the attribute
                    # in question and is great for informational purposes.

# Defining X and y variables
X = divorce
y = divorce[atr]

# Decision Tree Classifier
tree_clf = DecisionTreeClassifier(max_depth=6)
tree_clf.fit(X, y)

# Export tree using GraphViz
export_graphviz(
tree_clf,
out_file="divorce.dot")

# I used Graphviz Online to convert my .dot file to an image
# https://dreampuf.github.io/GraphvizOnline/