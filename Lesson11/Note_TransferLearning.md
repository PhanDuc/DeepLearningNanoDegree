# Transer Learning

Use pre-trained network and do new tasks. Add your own architecture on top of the pre-trained network.

### VGG network architecture

(conv+conv+max-pool) x 5 + (fc + relu)

```python
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

with tf.Session() as sess:
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_) #only build once
        
#only train relu6 layer
    feed_dict = {input_: images}
    codes = sess.run(vgg.relu6, feed_dict=feed_dict)
```

- one-hot code `scikit-learn::LabelBinarizer`

```python
lb = LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)
```

- shuffle data `scikit-learn::StratifiedShuffleSplit`

```python
ss = StratifiedShuffleSplit(n_splits = 1, test_size=0.2)
train_idx, val_idx = next(ss.split(codes, labels))
#split the validation in two validation and test
```

## General Idea

1. have a pre-trained network
2. Feed raw input to the pre-trained network
3. Get the layer output from the network 
    - e.g. get the output layer of the network
4. use that as the input to a new network (which you build)
5. and train your own network
6. In essense, you are building more layers on top up the original network.