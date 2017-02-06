# Learn to frame the problem

## Prepare for the Class
1. Activate a python3 conda environment. In this environment, you'll need to have 
installed `numpy`, `jupyter notebook`, `matplotlib`, `scikit-learn`, and `bokeh`.
    - create environment from **YAML** file: `conda env create -f environment.yaml`
    - create environment directly
        - `conda create -n env_name python=2` or `conda create -n env_name python=3`
        - `source activate env_name` (for Mac and Linux)
        - `conda install package_1 package_2 ... package_n`
        - `conda env export >environment.yaml` (export environment file for future use)
    - when leave the environment `source deactivate env_name`
2. Download and unzip the file at the bottom of this page.
3. Change directories into the unzipped folder.
4. Start up your Jupyter notebook server.
5. Open Sentiment Classification - Intro.ipynb.

## Introduction

### Framing the problem

Understand what is the input and output of the problem:

- Input: review
- output: label: [Negative, Position]

## Mini Project 1

### Topic: understand the data and frame the problem

### Ideas

Here, the idea is to find a measurement that can quantify the information -- information that tells us wheter the review is *Negative* or *Positive*

The ideas in solution

1. Count words in Positive reviews
2. Count words in Negative reviews
3. Calculate the ratio of each words in Pos v.s. Neg reviews
4. Show whether some words are mostly used in Positive reviews and other words are mostly used in Negative reviews

### Interesting code

```python
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
```
- Here the `lambda x:x[:-1].upper()` reads in the results of `g.readlines()`, and feed the result as `x` in the *lambda* function , and change every character into the upper case.

```python
from collections import Counter
```
- `collections` is the **high-performance container datatypes** in python. This seems to be an advanced version or `tuple`, `dict`, `list`, etc.
- `Counter` is the *dict subclass for counting hashable objects*
- `Counter.most_common()` returns all elements in the counter. We can specify `Counter.most_common([n])` to specify the most common `n` elements. 
    - e.g. `myCounter.most_common(3)`

## Mini Project 2

### Topic: Identify the input and output and quantify them

### Ideas

The structure of the neural network

- Input:
    - select words as input layers
    - count how many times each word in the review appears 
    - result in a vocabulary-count dictionary that represents the review text
    - because each note in input-layer is a word in the vocabulary, for a review text, its input quantitiy for each node is the count of such word appears in the text.
- Output: 
    - [0 / 1] for [Negative / Positive]
    - frame the problem as a **binary classification problem**

### Interesting code

```python
global layer_0
```
- Define global variable is ... a little uncommon


## Mini Project 3

### Topic: Build the neural network


### Ideas

Start to build the neural network

- The `SentimentNetwork` class is interesting, the code is good to learn

### Intersting code

The neural network class is very interesting. 


## Mini Project 4

### Topic: identify and reduce the input noise -- not important words


### Ideas

There are lots of noise in the input
    
- stopping words: e.g. *the*, *a*
- comma, period, and other markers.
- empty string

How to reduce the noise

- for each word: we only care about *exist or not*, so we don't count how many times a word appears
    - In this way, we can lower the input weight of the noise
    - At the same time, we make the real signals (e.g "excellent") become more important

### Intersting code

Nothing really interesting here

## Mini Project 5

### Topic: optimize the computation, sparse matrix operation

### Ideas

Each update is too slow. We want to optimize the computation speed of the neural network.

We have many *0* node that does not contribute any quantity to the next layer, but consumes a lot of computations time in matrix computations.

- Sparse Matrix Computation
    - DO NOT do the whole `.dot()` operation
    - Only updates from the nodes that are non-zero

### Intersting code

- Read in Train_data_raw, and preprocessing the train_data_raw into an index of existing words
- `self.layer_0` and `self.layer_1` has been initialized at the beginning.
- Also, the update of the weights also changed: only update the weights for those exsiting words

## Mini Porject 6

### Topic: Identify the real important words that are signals (gold)

### Ideas

We have calculated the pos-vs-neg ratio for each words

- plot the histogram of the ratio distribution
- looks lie a Normal distribution
- we need those words that are away from $\log(ratio) = 1$

We also want to see words distribution

- plot word-frequency in the text
- some words appears really a lot fo times, such as "the"
- we don't want those most frequent words and most infrequent words

Results

- Speed: much faster
- Accuracy: not necessarily higher

### Intersting code

```python
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                if(total_counts[word] > min_count):
                    if(word in pos_neg_ratios.keys()):
                        if((pos_neg_ratios[word] >= polarity_cutoff) \
                        or (pos_neg_ratios[word] <= -polarity_cutoff)):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)
        self.review_vocab = list(review_vocab)

```

## Understand the weight

### Topic: we want to understand the weight of a trained network

### Idea

Words that have similar value (pos / neg) may contribute similarly in the neural network, in terms of weights

- we can find out each input nodes and its weights to the hidden layer
- we can compare two nodes whether their weights are similar or not
- we can use `dot product` to measure this similarity of two weight vectors 

### Interesting code

```python
for word in mlp_full.word2index.keys():
        most_similar[word] = np.dot( \
        mlp_full.weights_0_1[mlp_full.word2index[word]], \
        mlp_full.weights_0_1[mlp_full.word2index[focus]])
```

- This code read out each word's weight and compare the weight vector to that of the **focus** word. Interesting approach

**Visualization the weight space**

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(vectors_list)
```

- t-distributed Stochastic Neighbor Embedding (tSNE) algorithm
- this set of data visualization code is interesting