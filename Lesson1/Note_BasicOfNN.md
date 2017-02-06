There are some interesting application to try and get a feel of Deep Learning

# Logistic Regression

Draw a line -> minimize errors by the line -> gradient descent

# Logistic Regression and Neural Network 
-> Neural network: multiple lines
-> Each node could be a logistic regression -> finally use a node to process previous nodes
-> Neural network can be treated as a multiple compound of simple logistic regressions?

# Perceptron

Data, like test scores and grades, is fed into a network of interconnected nodes. These individual nodes are called **perceptrons** or neurons, and they are the basic unit of a neural network. *Each one looks at input data and decides how to categorize that data*. In the example above, the input either passes a threshold for grades and test scores or doesn't, and so the two categories are: yes (passed the threshold) and no (didn't pass the threshold). These categories then combine to form a decision -- for example, if both nodes produce a "yes" output, then this student gains admission into the university.

Weights: These weights start out as random values, and as the neural network network learns more about what kind of input data leads to a student being accepted into a university, the network adjusts the weights based on any errors in categorization that the previous weights resulted in. This is called training the neural network.

## Activation Function

Finally, the result of the perceptron's summation is turned into an output signal! This is done by feeding the linear combination into an activation function.

One of the simplest activation functions is the **Heaviside step function**. [Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function): is just a simple thresholding function

## AND, OR, NOT, XOR perceptions 

- XOR perceptions need multiple layers

# Simplest Neural Network

$f(x) = Sigmoid(x) = 1/(1 + e^{-x})$

It's gradient: $f'(x) = f(x)(1 - f(x))$

The sigmoid function is bounded between 0 and 1, and as an output can be interpreted as a probability for success. It turns out, again, **using a sigmoid as the activation function results in the same formulation as logistic regression.**

# Learning Weights

What if you want to perform an operation, such as predicting college admission, but don't know the correct weights? You'll need to learn the weights from example data, then use those weights to make the predictions.

Error(Loss Function): 

- sum of the squared errors (SSE): $E = \frac{1}{2} \sum_{\mu}\sum_{j} [y_j^{\mu} - \hat{y}_j^{\mu}] ^ 2$

$\hat{y}_j^{\mu} = f( \sum_i w_{ij}x_i^{\mu})$

Use Gradient Descent Method to minimize the Loss Functions

- caveat: local minimal
- use [momentum](http://sebastianruder.com/optimizing-gradient-descent/index.html#momentum)

## Data Cleanup
- Use *dummy variable* to represent categorical variable (e.g. rank)
- Use Pandas: `data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)`
- Drop Column: `data = data.drop('rank', axis=1)`

## Initialize weights

First, you'll need to initialize the weights. We want these to be small such that the input to the sigmoid is in the linear region near 0 and not squashed at the high and low ends. It's also important to initialize them randomly so that they all have different starting values and diverge, breaking symmetry.

`weights = np.random.normal(scale=1/n_features**-.5, size=n_features)`

## Notes on Numpy on Linear Algebra

- Column Vector
	- a row vector `arr`, it's transpose is `arr.T` is still row vector
	- a row vector to a column vector -> `arr[:, None]`
	- make a row vector to be a 2D matrix -> `np.array(features, ndmin=2)` 
		- becomes a (n x 1) matrix
		- `arr.T` can get into a colum vector

## Backpropagation

- output error: y - output # real - model 
- output-error gradient: error * output * (1 - output)
- hidden layer error gradient:
	- $\delta_{j}^{h} = \sum_k W_{jk}\delta_{k}^{h+1}f'(h_j)$


for the $j$th node at $h$th layer, its error (model prediction v.s. output feedback) 
$$\delta_{j}^{h} = \sum_k W_{jk}\delta_{k}^{h+1}f'(h_j)$$

where $h_j = W_{in}^{h}X_{in}^{h} + b$, that is the input for this layer

Given the $\delta_{j}^{h}$, we can calculate the gradient descent step:
$$\Delta w_{ij} = \eta \delta_j^h x_i^{h - 1}$$

where $i$ is the index for input, $x_i$ are the input in the input-nearer level.


# References

- [Yes, you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.mzs7qtz1b)
- [Video Explanation](https://www.youtube.com/watch?v=59Hbtz7XgjM)
- [Frank Chen's Documentary](https://vimeo.com/170189199)

# TensorFlow

## General Idea:

>TensorFlow also does its heavy lifting outside Python, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from Python, TensorFlow lets us describe a graph of interacting operations that run entirely outside Python. (Approaches like this can be seen in a few machine learning libraries.)

## Tensor

- Formally, tensors are multilinear maps from vector spaces to the real numbers ( vector space, and dual space) 
- A scaler, a vector, and a matrix, are all tensor
- Common to have fixed basis, so **a tensor can be represented as a multidimensional array of numbers**.

- [A Visual and Interactive Guide to the Basics of Neural Networks](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)
	- This is a quite interesting reference.
- TensorFlow Tutorials
	- [O'Reilly](https://www.oreilly.com/learning/hello-tensorflow)
	- [Play with MNIST](https://www.tensorflow.org/tutorials/mnist/beginners/)
	- [An introduction Lecture](https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf)