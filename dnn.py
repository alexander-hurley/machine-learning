# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:06:49 2019

@author: Alexander Hurley

ML 3 - Deep Neural Network

Dataset divorce.csv was taken from https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set
and modified by Alexander Hurley for use in the INFR 3700U final project. 

NOTE!   Visualizing the DNN effectively turned out to be a daunting task. To compensate for the minimal 
        graphics used for the DNN, I have included extra statistical graphics for the dataset showing 
        various correlations and distributions. The code for those graphs is commented out at the 
        bottom of this file.

"""

# Import required libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
import pandas as pd

# I had to manually point to my GraphViz installation as the default installs
# are broken on the current build via Anaconda for Windows 10
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

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
'''
# Split train and test sets 80/20
from sklearn.model_selection import train_test_split
train, test = train_test_split(divorce, test_size=0.2)#, random_state=42)
y_train = train.iloc[:,-1]
x_train = train.drop(['divorce'], axis=1)
y_test = test.iloc[:,-1]
x_test = test.drop(['divorce'], axis=1)

# Standard Scaler using split data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Sequential Model using exponential, sigmoid, tanh and hard_sigmoid activations
model = Sequential()
model.add(Dense(40, activation='exponential', input_dim=54))
model.add(Dropout(0.3))
model.add(Dense(40, activation='sigmoid'))
model.add(Dropout(0.3))
#model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='hard_sigmoid'))
model.add(Dropout(0.3))

# If you run this, please comment out lines 91-118
# Calculate accuracy and loss
# Compile using the adamax optimizer
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adamax', 
              metrics=['accuracy'])

# Train model with the divorce train set
model.fit(x_train, y_train, epochs=15, batch_size=1, verbose=0)# validation_split=0.2)

# Evaluate on test set
loss, acc = model.evaluate(x_test, y_test, batch_size=8)

# Create a physical model (simple)
plot_model(model, to_file='model1.png', show_shapes=False, show_layer_names=False)

# Create a physical model (detail)
plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=True)

# Prints the Model Summary
model.summary()

# Print output accuracy and loss
print('\nDNN Test Results: loss: {0:2.5f}, val_acc: {1:2.3f}'.format(loss, acc))
'''
'''
# Calculate MSE of model
# If you run this, please comment out lines 65-90

# Compile using the adamax optimizer
model.compile(loss='mse',
              optimizer='adamax', 
              metrics=['accuracy'])

# Train model with the divorce train set
model.fit(x_train, y_train, epochs=15, batch_size=1, verbose=0)# validation_split=0.2)

# Evaluate on test set
loss, acc = model.evaluate(x_test, y_test, batch_size=8)

# Create a physical model (simple)
plot_model(model, to_file='model4.png', show_shapes=False, show_layer_names=False)

# Create a physical model (detail)
plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True)

# Prints the Model Summary
model.summary()

# Print output MSE
print('\nDNN Test Results: MSE: {0:2.5f}'.format(loss))
print("\n\n\n")
'''


# The code below is used to generate statistical graphs for understanding the 
# dataset. Some of the more interesting attributes to test are:
#     atr1, atr42, atr43, atr46, atr54

import matplotlib.pyplot as plt

# Preparing the graphs
attribute = "atr47"
graphPrep = divorce[[attribute, "divorce"]]

# Dataframe where divorce occurs
divy = graphPrep[graphPrep["divorce"] == 1]

# Plotting y for divorce
y1 = divy.atr47.value_counts(dropna=False).sort_index()

# Dataframe where divorce does not occur
divn = graphPrep[graphPrep["divorce"] == 0]

# Plotting y for no divorce
y2 = divn.atr47.value_counts(dropna=False).sort_index()

print(y1)
print(y2)

# Building the divorce bar graph as red
plt.bar(["1","2","3","4","5"], y1, color='red')
plt.show()

# Building the no divorce bar graph as blue
plt.bar(["1","2","3","4","5"], y2, color='blue')
plt.show()
