# Preparation

## Bag of Words

We only count how many times each word appears in the text. 

**We don't care about the order of the words in this model**

```python
from collections import Counter

def bag_of_words(text):
    # TODO: Implement bag of words
    return Counter(text.split()))
```

## Coverting Documents to Vectors

The idea is give each word a number ID, and the whole document becomes a vector of IDs.

## Word2vec

One of the most popular methods for creating these **word embeddings** is **Word2vec**. Word2vec is a neural network model that trains on text to create embeddings.

- Continuous bag of words (CBOW) 
- Skip grams
    - The skip grams model is a neural network that takes in a word and tries to predict the **n number of surrounding words**.

The input to the neural network: **jump**
The prediction of the neural network: n surrounding words, one at a time. [fox **jump** over]
The neural network will learn this surrounding information.

## RNNs and LSTMs Readings

- Christopher Olah's [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) on RNNs and LSTMs.
    - This is the shortest and most accessible read.
    - READING: N.A.
- [Deep Learning Book chapter on RNNs](http://www.deeplearningbook.org/contents/rnn.html).
    - This will be a very technical read and is recommended for students very comfortable with advanced mathematical notation and scientific papers.
    - READING: N.A.
- Andrej Karpathy's [lecture](https://www.youtube.com/watch?v=iX5V1WpxxkY) on Recurrent Neural Networks.
    - This is a fairly long lecture (around an hour) but covers the content quite well as always with Karpathy.
    - READING: N.A.

# Topic: Sentiment Analysis

- Lexicon-based
	- Tokenization - bags of words
- Machine Learning
	- Deep Learning captures the subtlety

## Readings

- Amazone AMI
	- [Tutorial](http://www.bitfusion.io/2016/05/09/easy-tensorflow-model-training-aws/)
		- READING: N.A.
- Sentiment Analysis
	- [LSTM on Sentiment](http://deeplearning.net/tutorial/lstm.html)
		- READING: N.A.
	- [Quora Answers](https://www.quora.com/How-is-deep-learning-used-in-sentiment-analysis)
		- READING: N.A.
	- [Deep Learning Sentiment One Character at a T-i-m-e](https://gab41.lab41.org/deep-learning-sentiment-one-character-at-a-t-i-m-e-6cd96e4f780d#.y6x98ycej)
		- READING: N.A.
	- [LSTM](http://k8si.github.io/2016/01/28/lstm-networks-for-sentiment-analysis-on-tweets.html)
		- READING: N.A.
- Bag of words
	- [Tutorials](https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words)
		- READING: N.A.
- Theoretical Background
	- [Lexicon-based](https://www.aclweb.org/anthology/J/J11/J11-2001.pdf)
		- READING: N.A.
	- [Lexicon-based v.s. Machine Learning](http://ceur-ws.org/Vol-1314/paper-06.pdf)
		- READING: N.A.