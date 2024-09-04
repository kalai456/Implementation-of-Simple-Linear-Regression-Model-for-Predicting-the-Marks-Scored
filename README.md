# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error,mean_squared_error
#read csv file
df=pd.read_csv(r'C:\Users\kalai\Downloads\student_scores.csv')
#displaying the content in datafile
df.head()

```
![image](https://github.com/user-attachments/assets/b819bfbf-47ef-4a7f-ac18-6c539b7e9a2e)
```
#segregating data to variables
x=df.iloc[:,:-1].values
x
```
![image](https://github.com/user-attachments/assets/231906a4-4438-4517-9860-956625a0ccd4)
```
y=df.iloc[:,1].values
y
```
![image](https://github.com/user-attachments/assets/d677363a-94b3-4366-8193-1039dba98e8d)
```
#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
```
```
#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
```
```
#displaying the predicted values
y_pred
```
![image](https://github.com/user-attachments/assets/6435bbe2-7c3b-4814-8abc-f672a6b67636)
```
#displaying actual values
y_test
```
![image](https://github.com/user-attachments/assets/f542ffdc-0e45-4647-bc43-33c010817749)
```
#graph plot for training data
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,reg.predict(x_train))
plt.title("hours vs marks")
plt.ylabel("marks obtained")
plt.xlabel("hours studied")
```
![image](https://github.com/user-attachments/assets/953024fe-3f15-4201-8d05-9bd3c59d5a0a)
```
#find mae,mse,rmse
mae=mean_absolute_error(y_test,y_pred)
mae
```
![image](https://github.com/user-attachments/assets/ed4aecfd-3bd3-4b75-836a-df4eb8919f03)
```
#Calculate Mean Absolute Error (MAE) and Mean Squared Error
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
![image](https://github.com/user-attachments/assets/187c79bc-7929-4544-96f9-6b0db4c87375)







## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
