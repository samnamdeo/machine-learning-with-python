#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#Encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#avoiding the dummy variable
x=x[:,1:]

#splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#fitting multiple linear Regression to the training set
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)

#predicting the test set
y_pred=regression.predict(x_test)

#visualizing the training sets
"""#plt.scatter(x_train,y_train,color="red")
#plt.plot(x_train,regression.predict(x_train),color="blue")
#plt.title("expences vs profit(training set)")
#plt.xlabel("expences")
#plt.ylabel("profit")
#plt.show()"""

#building a optimal model using backward elimination
import statsmodels.formula.api as sm

x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

x_opt=x[:,[0,1,2,3,4,5]]
regression_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regression_OLS.summary()

x_opt=x[:,[0,1,3,4,5]]
regression_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regression_OLS.summary()

x_opt=x[:,[0,3,4,5]]
regression_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regression_OLS.summary()

x_opt=x[:,[0,3,5]]
regression_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regression_OLS.summary()

x_opt=x[:,[0,3]]
regression_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regression_OLS.summary()












