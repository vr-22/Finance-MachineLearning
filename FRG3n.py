#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

data=pd.read_csv("Task 3 and 4_Loan_Data.csv")

data["debt/income"]=data["total_debt_outstanding"]/data["income"]
data["loan/income"]=data["loan_amt_outstanding"]/data["income"]
x=data[["credit_lines_outstanding","debt/income","loan/income","years_employed","fico_score"]]

def scale_down(x):
    return x.max(),x/x.max()
max_arr,x=scale_down(x)
y=data["default"]


# In[17]:


norml1=tf.keras.layers.Normalization(axis=-1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

norml1.adapt(x_train)
xn_train=norml1(x_train)

norml2=tf.keras.layers.Normalization(axis=-1)

norml2.adapt(x_test)
xn_test=norml2(x_test)


tf.random.set_seed(123)
model=Sequential([Dense(10,activation="relu"),
                  Dense(1,activation="sigmoid")])

model.compile(optimizer="adam",loss=tf.keras.losses.BinaryCrossentropy(),metrics=['binary_accuracy'])
model.fit(xn_train,y_train,epochs=80)


# In[19]:


y_predict=model.predict(xn_test)
model.evaluate(xn_test,y_test)


# In[12]:


def predict_default(inp):
    # Reshape the input to be a batch of size 1
    inp = inp.reshape(1, -1)

    # Use the previously adapted normalization layer for training data
    inp = norml1(inp)

    # Transform using the same PCA object used in training

    outcome = model.predict(inp)
    return outcome

credit_lines_outstanding=int(input("Enter credit_lines_outstanding :"))
loan_amt_outstanding=float(input("Enter loan_amt_outstanding :"))
total_debt_outstanding=float(input("Enter total_debt_outstanding :"))
income=float(input("Enter income :"))
years_employed=int(input("Enter years_employed :"))
fico_score=int(input("Enter fico_score :"))
inp=np.array([credit_lines_outstanding,loan_amt_outstanding/income,total_debt_outstanding/income,years_employed,fico_score])
inp/=max_arr
outcome=predict_default(np.array(inp))
print("The Probabalility of default is : ",outcome)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




