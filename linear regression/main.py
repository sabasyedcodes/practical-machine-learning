#predicting and training the model



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# creating our own data
# these values in horizontal format
# for sklearn we need vertical formatso reshape
time_studied = np.array([20,50,32,65,23,43,10,5,22,35,29,5,56]).reshape(-1,1)
scores = np.array([56,83,47,93,47,82,45,78,55,67,57,4,12]).reshape(-1,1)

#print(time_studied)

#Train the model
model = LinearRegression()
model.fit(time_studied,scores)

print(model.predict(np.array([56]).reshape(-1,1)))

plt.scatter(time_studied,scores)
#regression line
plt.plot(np.linspace(0,70,100).reshape(-1,1),model.predict(np.linspace(0,70,100).reshape(-1,1)),'r')
plt.ylim(0,100)
plt.show()


