#  Generative Adversarial Networks

## Games, Equilibrium, GANs Solution Render

Equilibirum: when neither player can improve their profit.

- GAN
    - generator minimize the cost
    - discriminator maximize the cost
    - arrive at a saddle point
    - equilibirum density -- true data density
- Common Failure    
    - When there are multiple clusters: Generator learns to generate 1 cluster not generate n cluster equally.

## Tricks and Tips

Choose a good architecture (At least one hidden layer)

- MNIST
    - Generator: input -> matmul + lrelu (leaky relu) -> matmul + tanh
    - Discriminator: matmul + lrelu -> matmul + sigmoid
- Loss
    - g_loss: AdamOptimizer(g_loss)
        - g_loss = cross_entropy(logits, flipped_labels)
    - d_loss: AdamOptimizer(d_loss)
        - d_loss = cross_entropy(logits, lebals)
    - overall: loss = sigmoid_cross_entropy(D_out, labels); label: 1==REAL; 0==FAKE
- Numerically stable cross
    - Loss = cross_entropy(logits, labels * 0.9)  #Use before the sigmoid; # *0.9 is GAN specific
- Negative d_loss as g_loss is not a good choice
- Scale up: convolution GAN
    - generator: input -> reshape -> convolution + lrelu -> convolution + lrelu -> ....
- Batch normalization is important

## Notebook Implementation

## Resource

- [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)