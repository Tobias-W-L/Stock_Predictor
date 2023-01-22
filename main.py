import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# https://www.section.io/engineering-education/stock-price-prediction-using-python/
# get features; following geeks features
ntflx = pd.read_csv('NFLX.csv')
splitted = ntflx['Date'].str.split('-', expand=True)

# organize all features and data
ntflx['day'] = splitted[2].astype('int')
ntflx['month'] = splitted[1].astype('int')
ntflx['year'] = splitted[0].astype('int')

ntflx['is_quarter_end'] = np.where(ntflx['month']%3==0,1,0)
ntflx['open-close'] = ntflx['Open'] - ntflx['Close']
ntflx['low-high'] = ntflx['Low'] - ntflx['High']
ntflx['target'] = np.where(ntflx['Close'].shift(-1) > ntflx['Close'], 1, 0)
features = ntflx[['low-high', 'open-close', 'is_quarter_end', 'Volume', 'target', 'Close']]

# use standard scale
scaler = StandardScaler()
# split data into test and train, then numpy them
# remember y_test for later graphing
data_train = np.array(features[:int(ntflx.shape[0]*0.8)])
data_test = np.array(features[int(ntflx.shape[0]*0.8):])

print(data_test.shape)
y_test_scaled = np.array(data_test[50:data_test.shape[0],5]).reshape((-1,1))

# scale each columns data in individual pairs 
# by creating new 2D numpy arrays for each column
# stack into new scaled datasets
scaled_train = np.array(data_train[:,0]).reshape((-1, 1))
scaled_test = new_test = np.array(data_test[:,0]).reshape((-1,1))
for x in range(1, 6):
	new_train = np.array(data_train[:,x]).reshape((-1, 1))
	new_test = np.array(data_test[:,x]).reshape((-1,1))
	new_train = scaler.fit_transform(new_train)
	new_test = scaler.transform(new_test)
	scaled_test = np.hstack((scaled_test, new_test))
	scaled_train = np.hstack((scaled_train, new_train))

data_train = scaled_train
data_test = scaled_test

# make x/y test/train arrays from normalized data
# first 50 rows of first 5 columns in x;
# last row of last column in y
def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0:5])
        y.append(df[i, 5])
    x = np.array(x)
    y = np.array(y)
    return x,y


x_test, y_test = create_dataset(data_test)
x_train, y_train = create_dataset(data_train)

model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 5)))
# dropout randomly removes neurons to reduce overfitting
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
# dense layer 1 outputs one value as final output
model.add(Dense(units=1))

# loss function 
model.compile(loss='mean_squared_error', optimizer='adam')
# model training
model.fit(x_train, y_train, epochs=50, batch_size=32)
model.save('stock_prediction.h5')
model = load_model('stock_prediction.h5')
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

fig, ax = plt.subplots(figsize=(16,8))
ax.set_facecolor('#000041')
ax.plot(y_test_scaled, color='red', label='Original price')
plt.plot(predictions, color='cyan', label='Predicted price')
plt.legend()
plt.show()