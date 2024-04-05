# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Kabilan V
RegisterNumber:  212222100018
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")





```

## Output:
## profit prediction
![image](https://github.com/kabilan22000284/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123469171/0ab53209-02a5-4a56-9943-90eab6853e2b)
## Function
![image](https://github.com/kabilan22000284/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123469171/0fb1c420-fe5f-4e33-8950-ae6afc86d145)
## Gradient descent
![image](https://github.com/kabilan22000284/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123469171/c3255500-128c-4e30-b037-9c414cb4c9c0)
## COST FUNCTION USING GRADIENT DESCENT:
![image](https://github.com/kabilan22000284/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123469171/7061d5b0-a6aa-4855-9c90-fdec3cbb12fa)
## LINEAR REGRESSION USING PROFIT PREDICTION:
![image](https://github.com/kabilan22000284/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123469171/fc6e44b4-1341-4f78-bfcb-09fb6657930c)
## PROFIT PREDICTION FOR A POPULATION OF 35000:
![image](https://github.com/kabilan22000284/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123469171/0d145671-a3c0-42f0-afd0-6cc69f3b1a8f)
## PROFIT PREDICTION FOR A POPULATION OF 70000:
![image](https://github.com/kabilan22000284/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123469171/6c30dfa9-cfb4-4fd5-a151-497ffb14a094)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
