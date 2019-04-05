import numpy as np 
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation
from tensorflow.contrib.keras.api.keras.optimizers import SGD, Adam
#import matplotlib.pyplot as plt

data = np.loadtxt('sin.csv', delimiter = ',', unpack = True)
x = data[0]
y = data[1]
model = Sequential()
model.add(Dense(30, input_shape = (1, )))
model.add(Activation('sigmoid'))
model.add(Dense(40))
model.add(Activation('sigmoid'))
model.add(Dense(1))
sgd = Adam(lr = 0.1)
model.compile(loss = 'mean_squared_error', optimizer = sgd)
model.fit(x, y, epochs = 1000, batch_size = 20, verbose = 0)
print ('save model')
model.save('sin_model.h5')
predictions = model.predict(x)
print (np.mean(np.square(predictions - y)))
preds = model.predict(x)
plt.plot(x, y, 'b', x, preds, 'r--')
plt.show()
