# Rent Prediction of the Houses in Paris using Multiple Linear Regression

## a. Problem Description
- We will build a model to estimate the rent by explaining the most influential boroughs of Paris the French capital. 
- The used method is the Multiple Linear Regression .
- The dataset used for building the model is a Comma Separated Value(.csv) file that contains 828 records.
- The <b> Price , Surface and Aggrandisement </b> are the three columns that we are using to build a model upon. 
- [View the Dataset - House Price.csv]( 
https://github.com/yogeshcenation/Rent-Prediction-of-House/blob/ba90d9bd6b576f860840b1cca3509e27ad1686b3/House%20Price.csv
)
***
### 1) Understanding Multiple Linear Regression 

- Multiple linear regression refers to a statistical technique that is used to predict the outcome of a variable based on the value of two or more variables. 

- It is sometimes known simply as multiple regression, and it is an extension of linear regression. 

- The variable that we want to predict is known as the dependent variable, while the variables we use to predict the value of the dependent variable are known as independent or explanatory variables.

### 2) Result Vector Predicted
- The result vector predicted is the vector that contains the concatenation of the predicted values of the independent variables and the actual value of the dependent variable.
- The outputs of the model are shown in the following code snippet :
#### - Predicting the Test set results
```python
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```
Output:

[[ 2.13  3.  ]
     [ 2.08  0.  ]
     [ 2.09  0.  ]
     [ 1.76  1.  ]
     [ 2.12  3.  ]
     [ 2.1   2.  ]
     [ 2.21  1.  ]
     [ 2.21  4.  ]
     [ 2.06  3.  ]
     [ 2.1   2.  ]
     [ 2.14  4.  ]
     [ 2.23  4.  ]
     [ 1.97  3.  ]
     [ 2.22  4.  ]
     [ 2.34  4.  ]
     [ 1.85  0.  ]
     [ 2.17  1.  ]
     [ 2.07  3.  ]
     [ 2.23  3.  ]
     [ 2.2   4.  ]
     [ 2.05  3.  ]
     [ 2.3   1.  ]
     [ 2.13  1.  ]
     [ 1.98  2.  ]
     [ 2.02  0.  ]
     [ 2.21  4.  ]
     [ 2.09  0.  ]
     [ 2.17  3.  ]
     [ 2.09  3.  ]
     [ 2.16  2.  ]
     [ 2.1   3.  ]
     [ 2.11  4.  ]
     [ 1.88  1.  ]
     [ 2.07  2.  ]
     [ 2.14  2.  ]
     [ 2.06  2.  ]
     [ 2.16  2.  ]
     [ 2.09  1.  ]
     [ 2.1   3.  ]
     [ 2.08  2.  ]
     [ 1.81  0.  ]
     [ 2.13  0.  ]
     [ 2.15  3.  ]
     [ 2.11  0.  ]
     [ 2.24  4.  ]
     [ 2.17  3.  ]
     [ 2.29  4.  ]
     [ 2.13  1.  ]
     [ 2.23  4.  ]
     [ 2.14  3.  ]
     [ 2.22  3.  ]
     [ 2.22  1.  ]
     [ 2.08  2.  ]
     [ 2.11  2.  ]
     [ 2.23  0.  ]
     [ 1.75  0.  ]
     [ 2.09  3.  ]
     [ 2.11  2.  ]
     [ 2.14  3.  ]
     [ 2.19  4.  ]
     [ 2.19  1.  ]
     [ 1.92  0.  ]
     [ 2.2   2.  ]
     [ 2.01  3.  ]
     [ 2.09  2.  ]
     [ 2.16  2.  ]
     [ 2.22  2.  ]
     [ 2.19  3.  ]
     [ 2.05  3.  ]
     [ 2.16  2.  ]
     [ 2.05  0.  ]
     [ 1.98  1.  ]
     [ 2.28  4.  ]
     [ 2.12  3.  ]
     [ 2.18  4.  ]
     [ 2.24  4.  ]
     [ 2.17  1.  ]
     [ 2.15  1.  ]
     [ 2.1   1.  ]
     [ 2.19  1.  ]
     [ 2.24  0.  ]
     [ 2.22  1.  ]
     [ 2.09  0.  ]
     [ 2.19  3.  ]
     [ 2.1   2.  ]
     [ 2.18  1.  ]
     [ 1.76  1.  ]
     [ 2.21  2.  ]
     [ 2.22  3.  ]
     [ 2.13  2.  ]
     [ 2.23  2.  ]
     [ 2.13  4.  ]
     [ 2.19  0.  ]
     [ 2.15  0.  ]
     [ 2.15  4.  ]
     [ 2.08  2.  ]
     [ 2.17  4.  ]
     [ 2.1   3.  ]
     [ 2.04  0.  ]
     [ 2.14  2.  ]
     [ 2.17  0.  ]
     [ 2.16  0.  ]
     [ 2.13  4.  ]
     [ 2.13  1.  ]
     [ 2.07  2.  ]
     [ 0.3   0.  ]
     [ 2.2   2.  ]
     [ 2.22  4.  ]
     [ 2.23  4.  ]
     [ 2.04  3.  ]
     [ 2.01  2.  ]
     [ 1.89  3.  ]
     [ 2.16  2.  ]
     [ 2.24  2.  ]
     [ 2.06  2.  ]
     [ 1.98  1.  ]
     [ 2.13  3.  ]
     [ 2.13  4.  ]
     [ 2.22  5.  ]
     [ 2.04  3.  ]
     [ 2.12  1.  ]
     [ 1.96  3.  ]
     [ 2.25  2.  ]
     [ 1.74  3.  ]
     [ 2.17  3.  ]
     [-2.7   0.  ]
     [ 2.14  2.  ]
     [ 2.25  2.  ]
     [ 2.12  1.  ]
     [ 2.17  3.  ]
     [ 2.19  4.  ]
     [ 1.61  0.  ]
     [ 2.15  0.  ]
     [ 2.21  1.  ]
     [ 2.18  3.  ]
     [ 2.17  3.  ]
     [ 2.17  0.  ]
     [ 2.19  4.  ]
     [ 2.18  3.  ]
     [ 2.19  0.  ]
     [ 2.41  2.  ]
     [ 2.29  2.  ]
     [ 2.04  0.  ]
     [ 2.1   3.  ]
     [ 2.25  0.  ]
     [ 2.11  3.  ]
     [ 2.16  1.  ]
     [ 2.26  4.  ]
     [ 2.23  3.  ]
     [ 2.04  2.  ]
     [ 1.82  3.  ]
     [ 1.97  1.  ]
     [ 2.1   0.  ]
     [ 2.29  4.  ]
     [ 1.99  3.  ]
     [ 2.23  4.  ]
     [ 2.16  3.  ]
     [ 2.13  3.  ]
     [ 2.22  4.  ]
     [ 2.27  4.  ]
     [ 2.04  2.  ]
     [ 2.12  1.  ]
     [ 2.18  2.  ]
     [ 2.23  4.  ]
     [ 2.28  4.  ]
     [ 2.18  1.  ]]

#### Evaluating the model using the Root Mean Squared Error

```python
from sklearn.metrics import mean_squared_error, r2_score
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

```
Output:

    Mean Squared Error: 1.775236421551857
    Root Mean Squared Error: 1.3323799839204493
    R2 Score: 0.05101236066805759

### 3) For Viewing the Source Code 
Press [Here](https://github.com/yogeshcenation/Real-Estate-Price-Prediction/blob/a8abc8e9ad7305ded35c3e343430d2b0b1c6a482/Real%20Estate%20Price%20Prediction.ipynb) to view the complete code 

### 4) Built Using
1) <b>IDE</b> -  <b>PyCharm 2021.3.3 (Professional Edition) </b> from JetBrains
2) <b>Base Interpreter</b> - <b>Python 3.10 </b> with Jupyter Kernel
***

## b. Libraries Dependencies 
* numpy -  for working with numerical arrays 
* pandas -  for working with the dataset
* sklearn - used for various operations to get the desired  task
* matplotlib -  used for working with graphs

Python 2 and 3 both work for this.

```
NOTE:
1. Install these libraries before you start to work with your own projects.
2. Visit https://pip.pypa.io/en/stable/ to install any dependencies.
```
***

## c. Other things this repository comes with 
- **Important:** [Getting Help](Repository Markups/getting-help.md)
- [Contact Me](Repository Markups/contact-me.md)

### 1) How can I thank you for writing and sharing this project?

You can star this project. Starring is free for you, but it tells me
and other people that you like this project.

Go [here](https://github.com/yogeshcenation/Real-Estate-Price-Prediction
) if you aren't here
already and click the "Star" button in the top right corner. You will be
asked to create a GitHub account if you don't already have one.
***
## d. Credits 
I'm  [Yogesh S](https://github.com/yogeshcenation).  and I have coded the project, but other people have helped me with it.

<b> If you like my work, then :</b>

<a href="https://www.buymeacoffee.com/yogeshcenation" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50px" width="170px" ></a>
***
If you have trouble with this project please [tell me about
it,](Repository Markups/contact-me.md) and I'll make this project better. If you
like this project, please [give it a
star](README.md-how-can-i-thank-you-for-writing-and-sharing-this-project).
