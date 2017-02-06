detection -> classification:

use a classifier that classifed small patches of the image into two classes (pedestraian / no pedestrian)

ranking -> classification: 

to classify pairs of (<query>, <web page>) as relevant/not relevant. Also, only classify promising candidates

### Logistic Classifier:

$$y = WX + b$$

find $W, b$ 

y = [score1, score2, score3]

turn $y$ into probability: softmax function

$$s(y_i) = \frac{e^{y_i}}{\sum_j e^{y_j}}$$

```
"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
```

Note: 
- if you multiple the score by 10, `softmax(scores * 10)`, the result could either be close to 1 or be close to 0.
- if you divided the score by 10, `softmax(scores / 10)`, the result could be similar to uniform

### One Hot Encoding: [1, 0, 0, 0] for multiclass encoding for training data

### Cross-Entropy

S  = [0.7, 0.2, 0.1] #Score
L  = [1, 0, 0]       #Label

--> distance from S to L (Not symmetric)
$$D(S, L) = - \sum_i L_i \log(S_i)$$
	
## multinomial logistic classification	
logits[output] -> softmax(logits) -> scores -> cross-entropy[label]

--> minimize the Cross-Entropy by training

$Loss = \frac{1}{N} \sum_i^{N} D(S(WX_i + b), L_i)$ # i for each data

Gradient Descent w.r.t. $W, b$

## Numerical Stability
floating number imprecision

we hope to have our variables (pre-processing the data, each channel)

$E(x) = 0$, $Var(x_i) = Var(x_j)$

badly conditioned -- optimizer has to do a lot of searching for good solution
well conditioned  -- optimizer can easily find solution

e.g. Image: (r - 128)/128; (g - 128)/128; (b - 128)/128

## good initialization

draw the weights from a N(0, \sigma)

\sigma -> determine the peak of the softmax distribution. 
\sigma larger -> more peaky distribution
\sigma small -> uncertaint -> less peaky distribution

## measuring performance

-Training Set
-Validation Set
-Testing Set

Classifier will try to "memorize" the training set. We need to care about the generalization. 

Validate -> take a small part of training set as the validation set
The classifier still sees the validation set indirectly during your tunning parameters

Still possible to overfit the model by testing on the validation set

Isolate the classifier from the test dataset --> hide a proportion that the classification never sees during the parameter-tunning process

The bigger the test set, the better to prevent overfitting. 
**Rule of Thumb**: A change of 30 test cases in results is usually trustworthy

(We are not talking about cross-validation)

Validation set:
- usually hold > 30000 examples
- changes >0.1% in accuracy 
- not balanced: heuristics not good anymore

## Find the right loss function

#Stochastic Gradient Descent (SGD)

Compute the gradient descent with **a random fraction of samples in training dataset**. Use a small random proportion to calculate the descend direction --> not accurate, but fast

### So many hyperparameters
- Initial Learning Rate
- Learning Rate Decay
- Momentum
- Batch Size
- Weight Initialization
- [ADAgrad]
	- no worries about hyperparameters
	- less fast than pure SGD
	

#Practical:

-Inputs:
	- Mean = 0
	- Equal Variance, and Small variance
-Initial Weights
	- Random initialization
	- Mean = 0
	- Equal Variance, and Small variance

## Momentum
- Use the running average of the gradience: $M <- 0.9 M + \Delta L$
- When update, use $\alpha * M(w1, w2)$
## Learning Rate Decay
- $\alpha$ decrease as steps goes on.
- Plot the [Loss v.s. Steps] for different learnign rate. 