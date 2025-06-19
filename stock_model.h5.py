import yfinance as yf
import pandas as pd
from tensorflow.keras.models import load_model

# Download Apple stock data
data = yf.download('AAPL', start='2019-01-01', end='2024-12-31')

# Display the first 5 rows
print(data.head())
import numpy as np

# Moving Averages
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Relative Strength Index (RSI)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Create Target: 1 if next day's close is higher than today's
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop missing values (NaNs from indicators)
data.dropna(inplace=True)

# Show result
print(data[['Close', 'MA10', 'MA50', 'RSI', 'Target']].head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Select features and target
features = ['MA10', 'MA50', 'RSI']
X = data[features]
y = data['Target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # for binary classification
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n📈 Model Test Accuracy: {accuracy:.2f}")
import matplotlib.pyplot as plt

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.title('Stock Direction Prediction (0 = Down, 1 = Up)')
plt.xlabel('Test Sample')
plt.ylabel('Direction')
plt.legend()
plt.show()
model.save('stock_model.h5')
model = load_model('stock_model.h5')

