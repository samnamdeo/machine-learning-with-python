#simple linear regression

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#splitting dataset into train_data and test_data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=1/3,random_state=0)

#fitting simple lineae regression to the training sets
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)

#predicting the test set result
y_predict=regression.predict(x_test)

#visualization of training sets
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regression.predict(x_train),color='blue')
plt.title('Salary vs Experience(traning model)')
plt.xlabel("year of Experience")
plt.ylabel("Salary")
plt.show()

#visualization of test set
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regression.predict(x_train),color='blue')
plt.title('salary vs experience(test model)')
plt.xlabel('year of experience')
plt.ylabel('salary')
plt.show()

