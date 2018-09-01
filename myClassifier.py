# Following tensorflow tutorial: https://www.tensorflow.org/tutorials/layers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    # TODO: swap in image size
    input_width = 200
    input_height = 200
    input_channels = 1

    # [batch_sz, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, input_width, input_height, input_channels])

    #Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    s1 = 2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=s1)

    # Convolutional Layer #2
    f2 = 64
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=f2,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    s2 = 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=s2)

    # Dense Layer
    # Unwrap the conv image: 28x28 after 2 rounds of stride=2 pools, and filter channel output of 64

    pool2_width = input_width / (s1 * s2)
    pool2_height = input_height / (s1 * s2)
    pool2_channels = f2

    dense_input_length = int(pool2_width * pool2_height * pool2_channels)

    pool2_flat = tf.reshape(pool2, [-1, dense_input_length])
    # 1024 is a bit arbitrary (but efficient), and samples down a reasonable ratio
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN) # cool implementation

    # Logits Layer
    class_count = 10
    logits = tf.layers.dense(inputs=dropout, units=class_count)

    predictions = {
        #generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # For simple predictions...
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Calculate the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # A little surprised this is the default optimizer # TODO: consider a more advanced optimizer (ADAM is pretty standard)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def get_data_set():
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images # Retruns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once("./ProcessedImages/Bad/*.png"))

    image_reader = tf.WholeFileReader()

    _, image_file = image_reader.read(filename_queue)

    image = tf.image.decode_jpeg(image_file)

    train_data =
    train_labels =
    eval_data =
    eval_labels =

    return (train_data, train_labels, eval_data, eval_labels)

def main(unused_argv):

    # TODO: load my own data
    # Load training and eval/test data
    # good_images =
    # bad_images =
    (train_data, train_labels, eval_data, eval_labels) = get_data_set()

    # Create the Estimator
    model_dir = "/tmp/mnist_convnet_model"
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir ) # TODO: Output trained model to my directory

    # Set up logging for p[redictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    print("train_data.shape")
    print(train_data.shape)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True )
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=100,
        hooks=[logging_hook] )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()
