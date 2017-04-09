# Siraj's Chatbot

## Architectures

input -> Encoder -> context -> Decoder -> Output

input -> embedding -> lstm-Encoder -> lstm-Decoder -> Output

## Preprocessing

| | Retrieval-Based | Generative-Based |
|:--:|:-----:|:---:|
|closed domain | Rules-Based | Smart Machine |
|open domain | Impossible | General AI |

Rules-based model: easy to do
Generative model: difficulty and tricy

## Seq2Seq in Tensorflow

- `tf.nn`, which allows us to construct different kinds of RNNs
- `tf.contrib.rnn`, which defines a number of RNN cells (an RNN cell is a required parameter for the RNNs defined in tf.nn).
- `tf.contrib.seq2seq`, which contains seq2seq decoders and loss operations.

- Encoder: this is a `tf.nn.dynamic_rnn`.
- Decoder: this is a `tf.contrib.seq2seq.dynamic_rnn_decoder`

### Inputs

- Tokenize words
    - add <GO>, <End of Sentence>, <Unknow>, and <PAD> tokens as well
- For the training output
    - add <GO> at the begining and <EoS> at the end
- Batching: put the <PAD> padding to make the length of each batch
- Be careful for the input. See Tensorflow document

### Tensorflow Document

- [Tensorflow Document](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq)

## Resources

- [Tensorflow tutorial](https://github.com/ematvey/tensorflow-seq2seq-tutorials)
- [(Data Set)Cornell Movie--Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [TF Stanford Tutorial](https://github.com/chiphuyen/tf-stanford-tutorials/tree/master/assignments/chatbot)
- [Deep Learning for Chatbots](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/)
- [Seq to Seq talk](https://www.youtube.com/watch?v=G5RY_SUJih4)
