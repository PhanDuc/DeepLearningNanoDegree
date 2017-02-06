Linear Regression

`from sklearn.linear_model import LinearRegression`

1. Linear Regression Works Best When the Data is Linear

Only captures the linear relationship

2. Linear Regression is Sensitive to Outliers

The best fit line is sensitive to the outliers

3. Multidimensional Linear Model

# Math, and Fitting with Gradient Descent

$$Error_{(m,b)} = \frac{1}{N}\Sum_{i=1}^N(y_i - (mx_i + b))^2$$

To minimize the error function, go gradient descent

gradient w.r.t $m$:
$$\frac{\partial }{\partial m}Error = \frac{2}{N}\Sum_{i=1}^N(- x_i(y_i - (mx_i + b)))$$

gradient w.r.t $b$:
$$\frac{\partial }{\partial b}Error = \frac{2}{N}\Sum_{i=1}^N(- (y_i - (mx_i + b)))$$

**Note:**

- Here, we calculate gradient descent of the cost function by taking **all** input data into consideration. Therefore, there is a $\Sum_{i = 1}^N$ term in calculation. 
- We want to find a model that predicts all the data in the best, that's why we take all training data into account
- It is not really scalable. When the size of training data increase, the computation is really expensive.
