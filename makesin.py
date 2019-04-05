import numpy as np 

x = np.linspace(-3.3, 3.4, 40)
y = np.sin(x)
data = np.c_[x, y]
np.savetxt('sin.csv', data, delimiter = ',', header = "x,y")
print ("ok")
