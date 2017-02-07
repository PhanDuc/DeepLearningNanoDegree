### Hidden Layer, Errors and Gradients

Here, the homework requires to separate the gradients and errors.

The **Gradiant** seems to be the gradient of the activation function of a layer

- `output_grad = 1`
- `hidden_grad = sigmoid_derivative(hidden_output)`

The **Layer Errors** seems to be the errors that propogate from the output layer

- `output_error = (targets - final_outputs)`
    - This definition is actually not straightforward becuse the real one should be `-(targets - final_outputs)`
    - However, since when we update weightes, we are actually use `+=` instead of `-=`, we have to be fore careful about the sign
- `hidden_error = output_errors.dot(self.weights_hidden_to_output)`

The update of the weight

- `weights_hidden_to_output =`
    - `learning rate *`
    - `output_errors *`
    - `output_grad(=1) *`
    - `its_input(=hidden_output)` 
- `weights_input_to_hidden = `
    - `learning rate *`
    - `hidden_errors *`
    - `hidden_grad *`
    - `its_input(=inputs)`

### Number of Nodes in Hidden Layers

> A good rule of thumb for deciding the number of nodes is halfway in between the number of input and output nodes. Typically, for this example, between 10 and 30 nodes tends to work well.

- Related Resource
    - [Quora Questions](https://www.quora.com/How-do-I-decide-the-number-of-nodes-in-a-hidden-layer-of-a-neural-network)