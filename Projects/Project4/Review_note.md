
# Review Note

## API parameters

TensorFlow API has a lot of parameters. It is recommended that always use keyword arguments.

Keyword arguments increase the readability of your code. It also prevent potential miss placing of argument order. (I personally can never remember those orders of parameters)

## Neural Network

`input = tf.placeholder(tf.int32, [None, None], name='input')`

As a good Python practice, you should not shadow builtin namespace at all. input is a builtin function.

## Dropout Training and Inference

I am a little bit suspicious about this implementation.

`lstm_dropped_out = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)`

This means that both your infer and training layer will use dropout results.

**You should implement dropout in training function only. Infer with dropout is not good.**

## References

- [RNNs in Tensorflow](http://web.stanford.edu/class/cs20si/lectures/slides_11.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)
- [What’s the Difference Between Deep Learning Training and Inference?](https://blogs.nvidia.com/blog/2016/08/22/difference-deep-learning-training-inference-ai/)
- [What My Deep Model Doesn't Know...](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)

## Other Suggested Parameters

```
batch_size=512
RNN_size = 256
num_layers = 3 or 4
encoding_embedding_size, decoding_embedding_size around 100
learn rate 0.01-0.05.
```