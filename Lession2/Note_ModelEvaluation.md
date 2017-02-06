# Model Evaluation

## Regression and Classification

- Regression returns a continuous value
- Classification returns a state

## What makes a good model?

- Generalization
- Overfitting

### Training and Testing

- Training set: train the model
- Testing set: evaluate the model
- Better Training does not necessarily result in a good model
- Better Testing <--> Better model

```python
from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.25)
#X: input
#y: output
#test_size: the proportion of the dataset used as the testing set
```

> Golden Rule: Thou shalt never use your testing data for training

### Metrics

1. Confusion Matrix (Signal Detection Theory): 
	- False Positive (Wrong)
	- False Negative (Wrong)
	- True Positive (Correct)
	- True Negative (Correct)
2. Accuracy
	- correct : total 
	- $\frac{correct}{total}$
3. Regression Metrics
	- Mean Absolute Error: 
		- $\sum_i |y_i - \hat{y_i}|$
	- Mean Square Error: 
		- $\sum_i (y_i - \hat{y_i})^2$
	- $R^2$ error: 
		- $1 - \frac{MSE_{regression}}{MSE_{simple}}$
		- Here, simple means use $\bar{y}$ as prediction, instead of $\hat{y_i}$.

```python
from sklearn.linear_model import LinearRegression

#error metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#linear regression and fitting
classifier = LinearRegression()
classifier.fit(X,y)

#get predicts
guesses = classifier.predict(X)

#calculate the errors
error = mean_absolute_error(y, guesses)
error = mean_squared_error(y, guesses)
error = r2_score(y, guesses)
```

### Types of Errors

- Underfitting: too general -- high bias
	- Error due to bias
- Overfitting: too specific -- high variance
	- Error due to variance
- Bias-variance trade-off
	- Model Complexity Graph

### Cross Validation

- Three sets
	- Training: train the model
	- Cross Validation: evaluate the model
	- Testing: test the model
- K-fold Cross Validation
	- Break the data into $K$ buckets
	- Average the results to get final results

```python
from sklearn.model_selection import KFold

kf = KFold(12, 3, shuffle = True) # Randomize the data

for train_indices, test_indices in kf:
	print train_indices, test_indices
```