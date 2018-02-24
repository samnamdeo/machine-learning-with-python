#Decision tree Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting decision tree regression to dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#predicting a new result
y_pred=regressor.predict(6.5)

#visualizing the result
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid)),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('truth vs bluff (Decision Tree Regressor)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()