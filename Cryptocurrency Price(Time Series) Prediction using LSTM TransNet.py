#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LayerNormalization, LSTM, Dense, MultiHeadAttention, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import yfinance as yf

tf.random.set_seed(123)

def download_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    data = pd.DataFrame(data)
    return data[["Close"]]

ticker = input("Enter currency ticker: ")
data = download_data(ticker, "2015-11-20", "2024-01-01")
time_steps = 30

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, time_steps):
    x = []
    y = []
    for i in range(len(data) - time_steps):
        x.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)

x, y = create_sequences(data_scaled, time_steps)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the LSTM and Transformer model
input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))

# LSTM layer
lstm_out = LSTM(30, return_sequences=True)(input_layer)
lstm_out = LayerNormalization()(lstm_out)

# Transformer layer
multi_head_attention = MultiHeadAttention(num_heads=8, key_dim=8, dropout=0.14)(lstm_out, lstm_out)
multi_head_attention = LayerNormalization()(multi_head_attention)

# Global average pooling layer
global_average = GlobalAveragePooling1D()(multi_head_attention)

# Fully connected layers
dense_1 = Dense(64, activation='relu')(global_average)
dense_2 = Dense(64, activation='relu')(dense_1)
dense_3 = Dense(32, activation='relu')(dense_2)

# Output layer
output_layer = Dense(1)(dense_3)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model on the test data
loss, mse = model.evaluate(x_test, y_test)
print(f'Mean Squared Error on Test Data: {mse}')

# Future prediction
def predict_future(model, data, time_steps, future_steps):
    predictions = []
    current_input = data[-time_steps:].reshape((1, time_steps, data.shape[1]))
    
    for _ in range(future_steps):
        next_prediction = model.predict(current_input)
        predictions.append(next_prediction[0, 0])
        next_input = np.append(current_input[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)
        current_input = next_input
    
    return np.array(predictions)

# Number of future steps to predict
future_steps = 10
future_predictions = predict_future(model, data_scaled, time_steps, future_steps)

# Inverse transform the predictions
future_predictions_rescaled = scaler.inverse_transform(future_predictions.reshape(-1, 1))

print(f"Future predictions for the next {future_steps} days:")
print(future_predictions_rescaled)


# In[ ]:




