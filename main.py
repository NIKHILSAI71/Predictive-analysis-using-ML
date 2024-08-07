import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Assuming you have a CSV file with columns: Date, Open, High, Low, Close, Volume
data = pd.read_csv("your_stock_data.csv", index_col="Date", parse_dates=True)
# Select the 'Close' price for prediction
data = data['Close']

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(np.reshape(train_data, (-1, 1)))
test_data = scaler.transform(np.reshape(test_data, (-1, 1)))

# Function to create dataset (assuming features are in the first dimension)
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]  # Assuming features are in first dimension
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)

# Create training and testing datasets
look_back = 60
trainX, trainY = create_dataset(train_data, look_back)

# Reshape for samples, time steps, features
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# Create and fit the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=32)

# Create training and testing datasets (again for prediction)
look_back = 60
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error
from sklearn.metrics import mean_squared_error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# Shift train predictions for plotting
trainPredictPlot = np.empty_like(data).reshape(data.shape[0], -1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
trainPredictPlot[:] = np.nan
testPredictPlot = np.empty_like(data).reshape(data.shape[0], -1)
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict


# Plot baseline and predictions
data_array = data.values.reshape(-1, 1)  # Reshape for compatibility
inverse_transformed_data = scaler.inverse_transform(data_array)
plt.plot(inverse_transformed_data)
