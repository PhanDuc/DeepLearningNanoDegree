
# Review Note

## Preprocessing

- create_lookup_tables
    - try `set` instead of `Counter`
    - write efficient code

## Build the Network

- Embedding
    - Use uniform distribution to initialize
    - `embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))`
    - Or use the tensorflow layer directly
    - `tf.contrib.layers.embed_sequence(input_data, vocab_size, embed_dim)`
- Get Batches
- 
```python
#Mat's version
    n_batches = int(len(int_text) / (batch_size * seq_length))

    # Drop the last few characters to make only full batches
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(x_batches, y_batches)))
```

- Build NN
    - I think you might want to use more epochs, and try a larger batch size. You also might want to try changing the truncated_normal parameters in get_embed. One last thing to try is a smaller rnn_size (if you try 200, I think it should work. This has to do with the rnn_size/embed_dim in get_embed()).

- pick words
     - [np.random.choice](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html)
     - or

```python
total = np.cumsum(probabilities)
chooser = np.sum(probabilities) * np.random.rand(1)
predicted_word = int_to_vocab[int(np.searchsorted(total, chooser))]
```