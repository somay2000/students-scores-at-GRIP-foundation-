# students-scores-at-GRIP-foundation-
I predicted the hour per students scores

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("student_scores.csv")
print(data)
a=data.describe()
print(a)
data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs percentage')
plt.xlabel('Hours studied')
plt.ylabel('percentage Scores')
plt.show()
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(x_train,y_train)
print(linreg.intercept_)
print(linreg.coef_)
y_pred=linreg.predict(x_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df)
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
