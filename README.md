# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kalaiselvan J
RegisterNumber:  212223080022


```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
```
df.head()
```
![image](https://github.com/user-attachments/assets/b4b38907-6efa-4a5f-a0e4-47f15f0729e5)
```
df.tail()
```
![image](https://github.com/user-attachments/assets/503457c3-aca3-47aa-a49c-261088fa9138)
```
Array value of X
```
![image](https://github.com/user-attachments/assets/0daf29ea-913a-4eab-8c31-44f1411bad14)
```
Array value of Y
```
![image](https://github.com/user-attachments/assets/e3cd164a-f262-4edf-adc7-c03a2505605d)
```
Values of Y prediction
```
![image](https://github.com/user-attachments/assets/788569d4-6f0f-4de6-ba94-c28ea7abacfb)
```
Array values of Y test
```
![image](https://github.com/user-attachments/assets/45e8ef69-f624-4bf2-a7cd-27bc43ad6b54)
```
Training Set Graph
```
![image](https://github.com/user-attachments/assets/3c5ff8f8-6e2b-48be-91fd-efdf9cfce059)
```
Test Set Graph
```
![image](https://github.com/user-attachments/assets/ac6344fe-6131-481c-8d23-8d8c84a9d190)
```
Values of MSE, MAE and RMSE
```
![image](https://github.com/user-attachments/assets/3bd9c1a4-a328-4a0e-accf-ac8d67a02a96)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
