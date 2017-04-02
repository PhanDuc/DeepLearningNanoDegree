# Deep Net with TensorFlow

## Examples

- [There are some tensorflow examples](https://github.com/aymericdamien/TensorFlow-Examples)
- [Siraj's video](https://www.youtube.com/watch?v=APmF6qE3Vjc)

## Major Resources

- [Andrej Karpathy's CS231n](http://cs231n.github.io/)
- [Michael Neilsen's Book](http://neuralnetworksanddeeplearning.com/)
- [Goodfellow, Bengio, and Courville's Book](http://deeplearningbook.org/)

## Related Code

- `cnn.py`
- `convolution_layer.py`
- `MNIST.py`
- `ft_variable_naming.py`

## ReLU function

$$F(x) = max(0, x) $$ 


Use ``tf.nn.relu()`` in the Tensorflow

```python
# Hidden Layer with ReLU activation function
hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases)
hidden_layer = tf.nn.relu(hidden_layer)

output = tf.add(tf.matmul(hidden_layer, output_weights), output_biases)
```

Print Sessino results

```python
with tf.Session() as sess:
    #first, initialize variables
    sess.run(tf.global_variables_initializer()) 
    print(sess.run(logits))
```

## TensorFlow for Deep Net

#### Basic Variables

For the neural network to train on your data, you need the following float32 tensors:

- features
    - **Placeholder tensor** for feature data (train_features/valid_features/test_features)
- labels
    - **Placeholder tensor** for label data (train_labels/valid_labels/test_labels)
- weights
    - **Variable Tensor** with random numbers from a truncated normal distribution. See `tf.truncated_normal()` [documentation](https://www.tensorflow.org/api_docs/python/constant_op.html#truncated_normal) for help.
- biases
    - **Variable Tensor** with all zeros. See  `tf.zeros()` [documentation](https://www.tensorflow.org/api_docs/python/constant_op.html#zeros) for help.

#### More information

See `MNIST.py` for the details.

When **SAVE** and **LOAD** the model, the variable definition order (define weights first, define biases second v.s. define biases first then define biases second) is important to keep the same

- Variable definition order need to be the same
    - because tensorflow will give the variables some names like `variable`, or `variable_1`
- If not the same, **explicitly give names to variables**

## Regularization

> To reduce the overfitting

1. Looking at the validation set -- STOP training as soon as performance stops improving
2. L2-regularization: $$\text{Loss'}= \text{Loss} +\beta\frac{1}{2} \|w\|^2_2$$
3. Dropout: randomly set half of the activation to be zero: $p(activation := 0) = 0.5$ for each activation
    - Each learned representation may be destroyed randomly.
    - Force the Deep Net to learn **Redundant Information** in order to achieve the results
    - For the evaluation, 
        1. Scale the remaining half of the activation by 2 -- to make sure sum == 1
        2. take the expectation of each activation  
        
implementation `tf.nn.dropout()`

```python
keep_prob = tf.placeholder(tf.float32) # probability to keep units

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
```

`keep_prob` allows you to adjust the number of units to drop. In order to compensate for dropped units, `tf.nn.dropout()` multiplies all units that are kept (i.e. not dropped) by `1/keep_prob`.


## Convolutional NN

### Statistical Invariance

- Translational invariance 
- Weight Sharing
    - Image: convolutional NN
    - Sentences: embedding, RNN

## Convolution NN

> Sharing weights for different elements in the image

> Remember that the CNN isn't "programmed" to look for certain characteristics. Rather, it learns on its own which characteristics to notice.

- Padding
    - valid padding: 
        - start with the edge of the original image;
        - results would be smaller than the original image
    - same padding:
        - add 0 paddings to the original image (to extend to original ones)
        - results would be the same size of the original image
    - TensorFlow Padding Document: [Here](https://www.tensorflow.org/api_guides/python/nn#Convolution)
- Filter
    - Stride: The amount by which the filter slides is referred to as the 'stride'. The stride is a hyperparameter which you, the engineer, can tune.
    - What's important here is that we are grouping together adjacent pixels and treating them as a collective.
    - **Filter Depth**
        - It's common to have more than one filter. Different filters pick up different qualities of a patch
        - The amount of filters in a convolutional layer is called the *filter depth*
    - Connection
        - Choosing a filter depth of `k` connects each patch to `k` neurons in the next layer.
        - Multiple neurons can be useful because a patch can have multiple interesting characteristics that we want to capture.
- Dimensionality
    - Given our input layer has a volume of $W$, our filter has a volume ($height * width * depth$) of $F$, we have a stride of $S$, and a padding of $p$, the following formula gives us the volume of the next layer: $(W−F+2P)/S+1$.

```python
input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'VALID'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```

Use ` tf.nn.conv2d()` and `tf.nn.bias_add()`. 

## Visualize CNN

[Paper](http://www.matthewzeiler.com/pubs/arxive2013/eccv2014.pdf)

[Visualization Toolbox](https://www.youtube.com/watch?v=ghEmQSxT6tw)

## Architecture

- Pooling
    - max-pooling $y = max(x_i)$ 
        0. `tf.nn.max_pool()`
        1. parameter free
        2. often more accurate
        3. more expensive computations
        4. more hyper parameters (pooling size, pooling stride)
    - average-pooling $y = mean(x_i)$
    - Functions: 
        1. decrease the size of the output 
        2. prevent overfitting. 
        3. Preventing overfitting is a consequence of the reducing the output size, which in turn, reduces the number of parameters in future layers.
    - Recently, pooling layers have fallen out of favor. Some reasons are:  
        1. Recent datasets are so big and complex we're more concerned about underfitting.
        2. Dropout is a much better regularizer.
        3. **Pooling results in a loss of information.** Think about the max pooling operation as an example. We only keep the largest of n numbers, thereby disregarding n-1 numbers completely.
    - Dimensionality
        - `new_height = (input_height - filter_height)/S + 1`
        - `new_width = (input_width - filter_width)/S + 1`
- 1x1 convolution
    - have small non-linearity over the patch
    - bascially is **matrix multiplication**
- inception
    - a composition of 
        1. average-pooling -> 1x1 convolution
        2. 1x1 convolution
        3. 1x1 convolution -> 3x3 convolution
        4. 1x1 convolution -> 5x5 convolution
    - concatenate the results above

**max pooling** 

```python
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# Apply Max Pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

The `tf.nn.max_pool()` function performs max pooling with the `ksize` parameter as the size of the filter and the `strides` parameter as the length of the stride. 2x2 filters with a stride of 2x2 are common in practice.