import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

x_list = []
y_list = []

for i in range(1000):
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.convolve(x,y)[:100]
    
    x_new = np.concatenate((z,x))
    x_list.append(x_new)
    y_list.append(y)



x_list = np.array(x_list)
y_list = np.array(y_list)

X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.30, random_state=40)

model = Sequential()
model.add(Dense(200, activation='relu', input_dim=100))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='softmax'))


model.compile(optimizer= "adam", loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

