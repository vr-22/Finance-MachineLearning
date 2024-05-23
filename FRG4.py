#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize

# Suppress TensorFlow warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# Read the data
data = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Feature engineering
data["debt/income"] = data["total_debt_outstanding"] / data["income"]
data["loan/income"] = data["loan_amt_outstanding"] / data["income"]
x = data[["credit_lines_outstanding", "debt/income", "loan/income", "years_employed", "fico_score"]]

# Scale the features
def scale_down(x):
    return x.max(), x / x.max()

max_arr, x = scale_down(x)

# Target variable
y = data["default"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Normalize the data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(x_train)
xn_train = normalizer(x_train)
xn_test = normalizer(x_test)

# Build and train the neural network model
model = Sequential([
    Dense(6, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=['binary_accuracy'])
model.fit(xn_train, y_train, epochs=80, verbose=0)

# Evaluate the model
model.evaluate(xn_test, y_test)

# Predict probabilities of default
x.columns = x.columns.astype(str)
xn = normalizer(x)
data["prob_default"] = pd.DataFrame(model.predict(xn))

# Perform Gaussian mixture modeling for clustering
clusters = 7
gmm = GaussianMixture(n_components=clusters, random_state=42)
inp_clus = pd.concat([x["fico_score"] * 850, data["prob_default"]], axis=1)
gmm.fit(inp_clus)
cluster_assignments = gmm.predict(inp_clus)

# Visualize clustering results
plt.scatter(inp_clus["fico_score"], inp_clus["prob_default"], c=cluster_assignments, cmap="viridis")
plt.xlabel("FICO Scores")
plt.ylabel("Probability of Default")
plt.title("Clustering Results")
plt.show()

# Define log-likelihood function for optimization
def log_likelihood(boundaries, fico_scores, prob_default):
    num_buckets = len(boundaries) - 1
    likelihood = 0

    for i in range(num_buckets):
        mask = (fico_scores >= boundaries[i]) & (fico_scores < boundaries[i + 1])
        ni = np.sum(mask)
        ki = np.sum(prob_default[mask])
        pi = ki / ni if ni > 0 else 0.0

        if ni > 0 and 0 < pi < 1:
            likelihood += ki * np.log(pi) + (ni - ki) * np.log(1 - pi)

    return -likelihood

# Optimize boundaries for FICO scores below the threshold
optimized_boundaries_below = optimize_boundaries(below_threshold)

# Optimize boundaries for FICO scores above the threshold
optimized_boundaries_above = optimize_boundaries(above_threshold)

# Combine the optimized boundaries
optimized_boundaries = np.concatenate([optimized_boundaries_below, optimized_boundaries_above])

# Create buckets based on optimized boundaries
buckets = pd.cut(inp["fico_score"], bins=np.unique(optimized_boundaries), labels=False)

# Visualize the result
plt.scatter(inp["fico_score"], inp["prob_default"], c=buckets, cmap="viridis")
plt.xlabel("FICO Scores")
plt.ylabel("Probability of Default")
plt.title("FICO Score Buckets")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




