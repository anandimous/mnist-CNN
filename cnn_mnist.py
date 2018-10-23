from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  # ----------------------------------------------------------------------------------
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])  ## Input image 28x28 pixels represented as a tensor


  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  ''' Constructs a two-dimensional convolutional layer '''
  # ----------------------------------------------------------------------------------
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)    ## Uses ReLU to normalize the output here


  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  '''  Constructs a two-dimensional pooling layer using the max-pooling algorithm '''
  # ----------------------------------------------------------------------------------
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)    ## standard pooling applied with windows of 2x2 size & stride = 2 


  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  ''' In the context of CNN, a filter is a set of learnable weights which are learned using the backpropagation algorithm. You can think of each filter
  as storing a single template/pattern. When you convolve this filter across the corresponding input, you are basically trying to find out the similarity
  between the stored template and different locations in the input. '''
  # ----------------------------------------------------------------------------------
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)    ## similar to conv1 but extracts 64 filters


  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  ''' applies pooling again results in 50 percent reduction of width and height from conv2 '''
  # ----------------------------------------------------------------------------------
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  ''' converts the tensor (i.e. higher-dimensional data) into 1D batch of vectors or REALLY long single dimensional tensor '''
  # ----------------------------------------------------------------------------------
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    
  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  ''' applies the densely connected layer summation to prev. input | # of neurons = 1024 '''
  # ----------------------------------------------------------------------------------
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)


  # Add dropout operation; 0.6 probability that element will be kept
  ''' regularization by dropout meaning 40 percent of the elements will be randomly dropped during training '''
  ''' if a significant element is dropped, it will appear again in the next training iteration & vice-versa '''
  # ----------------------------------------------------------------------------------
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    
  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  ''' logits map probabilities to set of real numbers --> Probability of 0.5 corresponds to a logit of 0. Negative logit correspond
  to probabilities less than 0.5, positive to > 0.5 in the range [+inf, -inf] '''
  # ----------------------------------------------------------------------------------
  logits = tf.layers.dense(inputs=dropout, units=10)

    
  ''' Softmax is a function that maps [-inf, +inf] to [0, 1] similar as Sigmoid. But Softmax also normalizes the sum of the values(output vector) to be 1 '''
  ''' classes --> predicted digit from 0-9 '''
  ''' probabilities --> list of probabilities for each possibility i.e. [ P(0), P(1) .... P(9) ] '''
  # ----------------------------------------------------------------------------------
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }


  ''' takes in the predictions in form of a dictionary and returns an EstimatorSpec object '''
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


  # Configure the Training Op (for TRAIN mode)
  ''' applies gradient descent optimization to improve the neural nets performance based on error value '''
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


  # Add evaluation metrics (for EVAL mode)
  ''' evaluates & prints out the loss, accuracy & epochs/steps that have been run '''
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  ''' data --> raw pixels values for 55,000 images of hand-drawn digits '''
  train_data = mnist.train.images  # Returns np.array
  ''' label --> corresponding digit value for each image '''
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  ''' raw pixels values for 10,000 images of hand-drawn digits '''
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

 
    # Create the Estimator
  ''' initializes the CNN classifier to enable use of convolution techniques '''
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    
  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

    
  # Train the model
  ''' trains the model over 20000 runs with 100 examples used at each step '''
  ''' num_epochs as None means the the model will train until the specified number of steps is reached '''
  ''' training data is shuffled '''
  # ----------------------------------------------------------------------------------
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])


  # Evaluate the model and print results
  ''' epoch is set to 1 to limit the # of times the model runs , shuffle set to False so it iterates over data sequentially '''
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
    tf.app.run()
