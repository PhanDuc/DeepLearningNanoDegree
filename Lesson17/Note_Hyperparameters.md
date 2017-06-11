#  Hyper Parameters

## Optimizer Hyperparameters

###  Learnig rate

- good starting: 0.01 
- usual list: 0.1 - 0.000001, by 1/10th
- Validation error:
    - Decreasing too slow -- increase learning rate
    - Increasing during training -- decrease learning rate
- Learning rate decay and Adaptive Learning rate (and Tensorflow implementation)
    1. [Exponetial Decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay) in Tensorflow
    1. [Adam](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer); It's [Paper](https://arxiv.org/pdf/1412.6980.pdf)
    2. [Adagrad](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer); It's [Paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) and [Introduction](http://cs.stanford.edu/~ppasupat/a9online/uploads/proximal_notes.pdf)

### Minibatch size

- Online Training vs Batch size
- Usual size: 1, 2, 4, 8, 16, 32, 64, 128, 256 (32 is usually a good candidate)
- Size:
    1. Small: gradient calculation has noise -- could be helpful to prevent us from local minima
    2. Large: more efficient, but memory demanding
- Resource: 
    - [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228)

### Epoch

- Pay attention to validation error
- **early stoping**: stop training when validation error stops descreasing

## Model Hyperparameters

### Number of Hidden Units / Layers

- Number of hidden units
    - the more the better (larger capacity)
    - not too large (prevent overtraining)
- Layers
    - 3 layers > 2 layers
    - n layers is similar to 3 layers
    - CNN: deeper the better
    
> in practice it is often the case that 3-layer neural networks will outperform 2-layer nets, but going even deeper (4,5,6-layer) rarely helps much more. This is in stark contrast to Convolutional Networks, where depth has been found to be an extremely important component for a good recognition system (e.g. on order of 10 learnable layers)." -- Andrej Karpathy
 

- Model Capacity [Deep Learning book, chapter 5.2](http://www.deeplearningbook.org/contents/ml.html)

### RNN Hyperparameters

- cell type
    - LSTM: commonly used
    - GRU: similar to LSTM
    - choose a random subset of the data to test these two
- layers
    - 2 - 3 layers
    - sometimes up to 7 could be helpful
- embedding
    - size up to 200
    - size more than 50 may only have marginal effect

### References

- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
- [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533)
- [Deep Learning Practical Methodology](http://www.deeplearningbook.org/contents/guidelines.html)
- [How to choose a neural network's hyper-parameters?](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters)

- [How to Generate a Good Word Embedding?](https://arxiv.org/abs/1507.05523)
- [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228)
- [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078)