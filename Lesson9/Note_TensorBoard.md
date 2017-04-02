# TensorBoard

- Vedio: [Tutorial](https://youtu.be/eBbEDRsCmv4)
- Doc: [Tutorial](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
- Should be installed together with tensorflow


## Viewing Graphs

- Write out the log file

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('./logs/1', sess.graph)
    
#separate training and testing
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_train = tf.summary.FileWriter('./logs/training', sess.graph)
    file_test = tf.summary.FileWriter('./logs/testing') # no need to write graph for test
```

- View the tensorboard

```bash
#in bash
$ tensorboard --logdir [log directory: i.e.logs/1]
``` 
Then you can go to webpage

- Names are specified in the model definition 

## Name Scopes

In order to have a good visualization, we use **name scopes** to group nodes in one scope (e.g. all nodes in the same layer) together.

```python
with tf.name_scope('scope name'):
    # nodes  in this scope

#example
with tf.name_scope('predictions'):
    preds = tf.nn.softmax(logits, name='predictions')
```

## Inspecting Variables

In order to understand the variables values, we want to summarize the variables values in the model. 

We have already organized our model in different name scopes, now we can summarize variables within each scope using `tf.summary` 

- single value
    - `tf.summary.scalar('name', variable)`
- multiple numbers
    - `tf.summary.histogram('name', variable)`
- group all the summary together into one node
    - `merged = tf.summary.merge_all()`

```python
#add merged summary when you run the session
summary, batch_loss, new_state, _ = sess.run[model.merged, model.cost, model.final_state, model.optimizer], feed_dict = feed)
#write training summary to log
iteration = e * n_batches + b
train_writer.add_summary(summary, iteration)
```

## Hyperparameters

1. We can wrap the training to be a function that reads in `(model, epoch, file_writer)`
2. We can then grid search hyperparameters (loop through)
3. For each combination of hyperparameters, we create a log string, such as 
    - `'logs/4/lr={}, rl={},ru={}'.format(learning_rate, num_layers, lstm_size)`
4. load all different runs together. [just load the folder]