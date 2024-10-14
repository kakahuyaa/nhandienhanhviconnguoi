import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
bodyswing_df = pd.read_csv("lac_nguoi.txt")
handswing_df = pd.read_csv("vay_tay.txt")
X = []
y = []
no_of_timesteps = 10

dataset = bodyswing_df.iloc[:,1:].values

n_sample = len(dataset)
# lấy 10 dữ liệu liên tiếp làm 1 sample
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)

dataset = handswing_df.iloc[:,1:].values

n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)
exit()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential() 
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2]))) 
model.add(Dropout(0.2)) # Dropout để tránh overfitting
model.add(LSTM(units = 50, return_sequences = True)) 
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation="sigmoid")) # Sigmoid để output là xác suất

model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy") # Binary crossentropy vì đây là bài toán binary classification (0 hoặc 1)
 
model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test)) # Train model với 16 epochs và batch size là 32 (mỗi lần train thì lấy 32 dữ liệu)
model.save("model1.h5")


