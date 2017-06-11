
# Review Note

## Unit Test

- [Unite Test Introduction](http://docs.python-guide.org/en/latest/writing/tests/)  from [the Hitchhiker's Guide to Python](http://docs.python-guide.org/en/latest/)

## TensorFlow

- [Placeholder basics](https://stackoverflow.com/documentation/tensorflow/2952/placeholders#t=20170512180513354874)

### Discriminator

- discriminator as a sequence of `conv` layers.
- using `conv2d` with `strides` to **avoid making sparse gradients** instead of *max-pooling layers as they make the model unstable*.
- using `leaky_relu` and **avoiding ReLU for the same reason of avoiding sparse gradients** as `leaky_relu` allows gradients to flow backwards unimpeded.
- using `sigmoid` as output layer
- **BatchNorm** to **avoid "internal covariate shift"** as batch normalisation minimises the effect of weights and parameters in successive forward and back pass on the initial data normalisation done to make the data comparable across features.
    - [Review](https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0) on this issue
- **Use a smaller model for the discriminator relative to generator** as stated in this review of GANs as generation is a much harder task and requires more parameters, so the generator should be significantly bigger.
    - [Review](https://github.com/tensorflow/magenta/blob/master/magenta/reviews/GAN.md) on this issue
- Use **weight initialization**: `Xavier initialization` is recommended so as to break symmetry and thus, help converge faster as well as prevent local minima
    - [Blog Review](https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/) on this issue
    - [1] If the weights in a network start too small, then the signal shrinks as it passes through each layer until it’s too tiny to be useful.
    - [2] If the weights in a network start too large, then the signal grows as it passes through each layer until it’s too massive to be useful.
    - Xavier initialization makes sure the weights are ‘just right’, keeping the signal in a reasonable range of values through many layers.
    - A possible [implementation in tensorflow](https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers/initializers) is to pass `tf.contrib.layers.xavier_initializer()` as the value for the `kernel_initializer` paramter in `tf.layers.conv2d`
- Use `Dropouts` **in discriminator** so as to make it less prone to the mistakes the generator can exploit instead of learning the data distribution as mentioned [here](https://github.com/tensorflow/magenta/blob/master/magenta/reviews/GAN.md#disadvantages). Possible [tensorflow implementation](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) can be achieved by simply passing the outputs from the previous layer for each block (except the last) into the `tf.nn.dropout` with **a high `keep_probability`**.

### Generator

- `Tanh` as the last layer of the generator output. This means that we'll have to **normalise the input images** to be between -1 and 1.
- Used a *smaller model* for the **discriminator** relative to generator.
- Consider `Xavier initialization`
- Use Dropouts(50%) in generator in both [train and test phase](https://github.com/soumith/ganhacks#17-use-dropouts-in-g-in-both-train-and-test-phase) so as to provide noise and apply on several layers of the generator both training and test time. This was first introduced in an image translation paper called [pix2pix](https://arxiv.org/pdf/1611.07004v1.pdf) for which you can check out an awesome demo here.

### function `model_loss` is implemented correctly

- You did a fantastic job here as the loss function for GANs can be super convoluted and confusing. For more tips and hacks refer this [GAN Hacks link](https://github.com/soumith/ganhacks) by Soumith Chintala, one of the co-authors of the [DCGAN](https://arxiv.org/abs/1511.06434) as well as of [Wasserstein GAN](https://arxiv.org/abs/1701.07875).

- A small improvement here is also possible. To prevent **discriminator** from being too strong as well as to help it generalise better the discriminator labels are reduced from 1 to 0.9. This is called **label smoothing** (one-sided).
    - A possible TensorFlow implementation is `labels = tf.ones_like(tensor) * (1 - smooth)`

### Neural Network Training [A key part that I have missed]


- Fantastic job providing z between -1 and 1 and using `np.random.uniform(-1, 1, size=(batch_size, z_dim))`.

- **Normalising the inputs**: Since we are using `tanh` as the last layer of the generator output and *so the real image should also be normalized so that the input for the discriminator (be it from generator or the real image) lies within the same range*, we have to normalise the input images between -1 and 1.
    - Now I'd like you to figure out the current range (either by using the numpys' min and max or by carefully reading the data description given in the notebook) and then normalise it such that it lies between -1 and 1.

#### Hyper-parameter

For your network architecture, the choice of hyperparameters is mostly reasonable.

- Learning Rate: The current rate is very reasonable.
- Beta1: Your chosen value for beta1 is good.
- Good value of **alpha/ leak** parameter has been chosen. However, still try lowering to 0.1 .
- Z-dim: 100/128 is a reasonably good choice here.
- Batch Size: The chosen batch_size is appropriate (~32 to 64) because
    - If you choose a batch size too small then the gradients will become more unstable and would need to reduce the learning rate. **So batch size and learning rate are linked**.
    - Also if one use a batch size too big then the gradients will become less noisy but it will take longer to converge.
- a good value of **label smoothing** ~ 0.1 to prevent the discriminator from being too strong for the generator.
- Number of Filters/ depth of each layer: Good choice.