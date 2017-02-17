# Mini Flow

This is a project that helps to build a small tensorflow on our own.

## Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a version of Gradient Descent where on each forward pass a batch of data is randomly sampled from total dataset. Remember when we talked about the batch size earlier? That's the size of the batch. Ideally, the entire dataset would be fed into the neural network on each forward pass, but in practice, it's not practical due to memory constraints. SGD is an approximation of Gradient Descent, the more batches processed by the neural network, the better the approximation.

A naïve implementation of SGD involves:

- Randomly sample a batch of data from the total dataset.
- Running the network forward and backward to calculate the gradient (with data from (1)).
- Apply the gradient descent update.
- Repeat steps 1-3 until convergence or the loop is stopped by another mechanism (i.e. the number of epochs).


## Design

The Design of Mini Flow is quite interesting. There are multiple nodes.

- Input Nodes
    - No calculation within the node
    - No other inputs
    - DATA INPUT, WEIGHTS, BIASES are all Input nodes
- Linear Nodes
    - do `Linear Combination` only
    - Take inputs, weights, and biases as **inbound** and calculate the value
    - Only connect to next stage for the **outbound**
- Sigmoid Nodes
    - do `Sigmoid Transformation` only
    - Take only one input as **inbound**
    - value is the sigmoid transforamtion
    - connect to outbound
- etc, list MSE nodes to calculate the MSE
- So bascially, every operation has a node, every node only do one operation.

All the operations and computations are done on nodes. Nodes, however, only represents the structure of the neural network, and computations within the structure. **It does not concern any values**.

To give values to the structure, the MiniFlow used `topological_sort` to understand the structure and map values onto the structure.

Each node has its own

- Value
- Gradient
- in_bound (input)
- out_bound (output)

To update the value: `node.value -= learning_rate * node.gradient`. In this way, everything is cleaned and packed into different nodes.

The whole pipline of the information

1. Forward: from input-layer to output-layer, calculate the value
2. Backward: calculate the errors, and compute the partial derivatives
3. Update: for each node (weight and bias), update its value w.r.t. its particial derivatives in 2)

Because `value` and `gradients` are all saved in each node respectively, we can do the backward and update independently.