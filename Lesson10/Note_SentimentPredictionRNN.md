# Sentiment Prediction RNN

Word -> Embedding -> LSTM -> Sigmoid

If the passage is too long (2000+ words), the RNN needs really a long time to train the network.

- **lstm_size**: the number of units in the hidden layers in the LSTM cells. (kinda set numbers of units in hidden layer)
- **lstm_layers**: how many lstm layers in network. (start with 1)

## Embedding
Now we'll add an embedding layer. We need to do this because there are 74000 words in our vocabulary. It is massively inefficient to one-hot encode our classes here. You should remember dealing with this problem from the word2vec lesson. Instead of one-hot encoding, we can have an embedding layer and use that layer as a lookup table. You could train an embedding layer using word2vec, then load it here. But, it's fine to just make a new layer and let the network learn the weights.

**Exercise**: Create the embedding lookup matrix as a tf.Variable. Use that embedding matrix to get the embedded vectors to pass to the LSTM cell with `tf.nn.embedding_lookup`. This function takes the embedding matrix and an input tensor, such as the review vectors. Then, it'll return another tensor with the embedded vectors. So, if the embedding layer as 200 units, the function will return a tensor with size [batch_size, 200].

[tf.nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup): when cannot use one-hot(too inefficient), we try to regard each word as a vector of length n (word2vec)

```python
with graph.as_default():
#just get an embedding layer
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)
```

## LSTM Cell

- [Document](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn)

- Build LSTM

```python
with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) 
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)
```

- 