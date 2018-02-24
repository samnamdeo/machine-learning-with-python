# Polinomial linear regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting Simple Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#visualizing the linear regression result
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x))
plt.title('true vs false')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualizing the polynomial regression result
x_grid=np.arange(min(x),max(x),0.1) #for more smooth curve and remove straight line from PLR
x_grid=x_grid.reshape((len(x_grid),1))#reshape becaz x_grid is array and coverting it into a matrix
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('polynomial model')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predicting a new result with linear Regression model
lin_reg.predict(6.5)

#predicting a new result with Polinomial Regression model
lin_reg2.predict(poly_reg.fit_transform(6.5))







