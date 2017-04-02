# Weight Initialization

## Ones and Zeros

Simpliest idea: either all 0s, or all 1s. 

- **Worst Idea!**
- **Very bad for backpropagation**
- Cannot differentiate nodes and weights
- No variability
- All 1s
    - Loss is flat
- All 0s
    - Loss are really large

## Uniform Distribution

```python
tf.random.uniform(shape, nimval, maxval, dtype, seed, name)
```
show results compared with ones and zeros.

**General rule**: Best range for uniform distribution: $[-y, y], y = \sqrt{n}$, where $n$ is the number of inputs to a given neuron.

for example: `[-1, 1)` is much better than `[0,1)`. You can do experiments on this.

## Too Small

`[-1, 1)`  is worse than `[-0.1, 0.1)`, but when the range is even smaller, such as `[-0.01, 0.01)`, the loss increases and accuracy drops.

## Normal Distribution


```python
tf.random.normal(shape, mean, stddev, dtype, seed, name)
```

Discard values large than 2 standard deviation
```python
tf.random.truncated_normal(shape, mean, stddev, dtype, seed, name)
```

- Prefer truncated normal
- How to set the `stddev`
    - don't be too large
- much better than uniform distribution
    
## Additional Material

- [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
- [Delving Deep into Rectifiers:
Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852v1.pdf)
- [Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v2.pdf)