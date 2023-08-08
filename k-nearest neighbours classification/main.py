#we are using breast cancer dataset 

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np 

data = load_breast_cancer()
#parameters we can use to tell harmful or not harmful
#it's shape , size like basically features 
print(data.feature_names)
print(data.target_names)

#spliting the data into training data and testing data
# the split is gonna be random it does not take first 80 and last 20 , it shuffles
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data),np.array(data.target),test_size=0.2)
"""
The n_neighbors parameter in the KNeighborsClassifier refers to the number of nearest neighbors to consider when making predictions for a new data point. 
"""
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)

#evaluate the model 
print(clf.score(x_test,y_test))

'''
what we can do further is pass the data into  numpy array then pass to  predict method and then classify them either malignant or benign
'''
#clf.predict([])





