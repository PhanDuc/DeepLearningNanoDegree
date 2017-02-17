# Intro to TensorFlow

## TensorFlow Official Website

[Here we go](https://www.tensorflow.org/)

## Install Tensorflow in conda

`conda install -c conda-forge tensorflow`

## TensorFlow Intro 1

`import tensorflow as tf`

### Constant 

> Tensor: n-dimensional array

```python
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([ [123,456,789] ]) 
 # C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
```
### Session
> A **"TensorFlow Session"**,  is an environment for running a graph. The session is in charge of allocating the operations to GPU(s) and/or CPU(s), including remote machines.

```python
with tf.Session() as sess:
    output = sess.run(hello_constant)
```
### Placeholder
> **Tensorflow placeholder**: `tf.placeholder()` returns a tensor that gets its value from data passed to the `tf.session.run()` function, allowing you to set the input right before the session runs.

Use `feed_dict` to supply values for the placeholders

Example 1: ONE placeholder

```python
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```

Example 2: Multiple placeholders

```python
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
```

> Possible Error: If the data passed to the feed_dict doesn’t match the tensor type and can’t be cast into the tensor type, you’ll get the error “ValueError: invalid literal for...”.

### Variable

The `tf.Variable` class creates a tensor **with an initial value** that can be modified, much like a normal Python variable. This tensor stores its state in the session, so you must initialize the state of the tensor manually -- ` tf.global_variables_initializer() `. The `tf.global_variables_initializer()` call returns an operation that will initialize all TensorFlow variables from the graph.


**Initialization**

- initialize all variables:
    - `sess.run(tf.global_variables_initializer())`
- define a variable to be initialized with normal distribution
    - `tf.truncated_normal()`: return value in [mean - 2SD, mean + 2SD]
- define a variable to be initialized with zero value
    - `tf.zero()`

Example 1. Global variable initialization

```python
x = tf.Variable(5)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

Example 2. Initialization with normal distribution

```python
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
```

Example 3. Initialization with zeros

```python
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))
```

### Conclusion

It seems that the model definition part is independent of model running part. 

- Before the `session`, a model is defined, including how a variable needs to be initialized
- Before the `session`, some data is prepared, but it is separated from the model
- In the `session`, data was feeded into the model with `sess.run()`, where `feed_dict=` provide a dictionary and tell the system feed what data into what variables.

## TensorFlow Math Intro

- [Document](https://www.tensorflow.org/api_docs/python/math_ops/)

### Bascis

- add
- sub
- mul
- div
- matmul 
    - matrix multiplication
- reduce_sum
    - sum up an array of numbers and return one value
- log

```python
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.sub(tf.div(x,y),tf.constant(1))

# TODO: Print z from a session
# Always RUN to get results
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
```



## Logistic Regression

$$WX + b = y$$

or based on the dimensionality defined in $W$, we can write the above to b

$$y = xW + b$$

$W$ is the weights, $X$ is the input, $b$ is the bias

### Softmax Function

$$S(y_i) = \frac{\exp(y_i)}{\sum_j\exp(y_j)}$$

The **scores ($y_i$)** are often called **logits**

### Sofrmax in TensorFlow

`tf.nn.softmax()`

## One-Hot Coding (Dummy Variables)

a multi-class variable becomes **n variables** and each lable to be `1` for its class and `0` for everywhere else.

### sklearn implementation `LabelBinarizer`

```python
import numpy as np
from sklearn import preprocessing

# Example labels
labels = np.array([1,5,3,2,1,4,2,1,3])

# Create the encoder
lb = preprocessing.LabelBinarizer()

# Here the encoder finds the classes and assigns one-hot vectors 
lb.fit(labels)

# And finally, transform the labels into one-hot encoded vectors
lb.transform(labels)

>>> array([[1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0]])
```

## Numerical Stability

Adding a very small value to a very large value can lead to problems

To help ease the problem of numerical stability

1. Normalization So, for all factors $x_1, x_2, x_3, ... x_n$, ideally, we want $$E(x_i) = 0$$ $$Var(x_i) = Var(x_j)$$
2. Well conditioned v.s. Badly conditioned.
3. Initalized weight from a normal distribution
    - $\sigma$ of the normal distribution: use a less certain distribution

## Model Evaluation

- Training
- Validation
- Testing

## Stochastic Gradient Descent (S.G.D)

> Scaling the gradient descent solution

We only randomly sample a proportion of the whole dataset to train the model. We pretend that the gradient direction calculated from these random samples is the gradient direction of the model of all data

### Help S.G.D

1. Input: nomalization and equal variance
2. weight: initialization with normal distribution
3. Momentum
    - Gradient Direction := Running Average of the gradient
    - $M = 0.9 M + \Delta L$, where $M$ is momentum, and $L$ is the Loss 
4. Learning Rate Decay
    - gradually decrease learning rate
    - e,g, Exponential Decay
    - **Loss v.s. Steps** Figure to see how learning rate affect learning  
    - when learning is not good, lower your learning rate first
5. ADAGRAD
    - A SGD-variant that automatically use momentum and adaptive learning rate
    
### Mini-batching

Mini-batching is a technique for **training on subsets of the dataset instead of all the data** at one time. This provides the ability to train a model, even if a computer lacks the memory to store the entire dataset.

Mini-batching is computationally inefficient, since you can't calculate the loss simultaneously across all samples. However, this is a small price to pay in order to be able to run the model at all.

It's also quite useful combined with SGD. **The idea is to randomly shuffle the data at the start of each epoch, then create the mini-batches**. For each mini-batch, you train the network weights with gradient descent. Since these batches are random, you're performing SGD with each batch.

**Note:** you may not get mini-batch of equal size. For example, if you want to have a mini-batch of 128 in size, a dataset of 1000 data cannot be divided equally. So we want to have the flexibility of working with different size.

```python
# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
```

The `None` dimension is a placeholder for the batch size. At runtime, TensorFlow will accept any batch size greater than 0.

### Epoches

An epoch is a single forward and backward pass of the whole dataset. This is used to increase the accuracy of the model without requiring more data. 

The code for multi-epoch training. 

- Load Libraries

```python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from helper import batches  # Helper function created in Mini-batching section
```
- Define functions to print results

```python
def print_epoch_stats(epoch_i, sess, last_features, last_labels):
    """
    Print cost and validation accuracy of an epoch
    """
    current_cost = sess.run(
        cost,
        feed_dict={features: last_features, labels: last_labels})
    valid_accuracy = sess.run(
        accuracy,
        feed_dict={features: valid_features, labels: valid_labels})
    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        epoch_i,
        current_cost,
        valid_accuracy))
```

- Read in Data

```python
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)
```

- Define the Model

```python
# The features are already scaled and the data is shuffled
train_features = mnist.train.images
valid_features = mnist.validation.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
valid_labels = mnist.validation.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
- Run the model with data

```python
init = tf.global_variables_initializer()

batch_size = 128
epochs = 10
learn_rate = 0.001

train_batches = batches(batch_size, train_features, train_labels)

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch_i in range(epochs):

        # Loop over all batches
        for batch_features, batch_labels in train_batches:
            train_feed_dict = {
                features: batch_features,
                labels: batch_labels,
                learning_rate: learn_rate}
            sess.run(optimizer, feed_dict=train_feed_dict)

        # Print cost and validation accuracy of an epoch
        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))
```