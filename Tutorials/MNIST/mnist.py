import tensorflow as tf
import numpy as np
import argparse

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name="Weights")

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name="biases")

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


##model

def model(input):
    x_image = tf.reshape(input, [-1, 28, 28, 1])

    with tf.variable_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.variable_scope("maxpool1"):
        h_pool1 = max_pool_2x2(h_conv1)

    print("conv1", h_conv1.shape)
    print("pool1", h_pool1.shape)

    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.variable_scope("maxpool2"):
        h_pool2 = max_pool_2x2(h_conv2)

    print("conv2", h_conv2.shape)
    print("pool2", h_pool2.shape)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    print("pool2_flat", h_pool2_flat.shape)

    with tf.name_scope("fc"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    print("fc1", h_fc1.shape)

    keep_prob = tf.placeholder(tf.float32)
    tf.add_to_collection("keep_prob",keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope("output"):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    print("y", y_conv.shape)
    
    return y_conv,keep_prob


def main(args):
    
    ##Get the data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    ##Build graph
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    tf.add_to_collection("input",x)
    tf.add_to_collection("output",y_)

    y_conv,keep_prob = model(x)
    tf.add_to_collection("model_out",y_conv)

    ## Train branch
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    ce_summary = tf.summary.scalar("Cross_entropy_loss",cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    ##validation branch
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    ac_summary = tf.summary.scalar("accuracy",accuracy)
    tf.add_to_collection("acc",accuracy)

    ##Run the graph

    merged_ac = tf.summary.merge([ac_summary,])
    merged_ce = tf.summary.merge([ce_summary,])

    with tf.Session() as sess:

      writer = tf.summary.FileWriter(args.events_dir,sess.graph)

      sess.run(tf.global_variables_initializer())
      for i in range(3000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
          summary_ac, train_accuracy = sess.run([merged_ac,accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
          print('step %d, training accuracy %g' % (i, train_accuracy))
          writer.add_summary(summary_ac,i)
        summary_ce, _ = sess.run([merged_ce,train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        writer.add_summary(summary_ce,i)


      print('test accuracy %g' % accuracy.eval(feed_dict={
          x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

      saver = tf.train.Saver()
      saver.save(sess,args.model_dir)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/MNIST/'
    )
    
    parser.add_argument(
      '--events_dir',
      type=str,
      default='/tmp/MNIST/events'
    )
    
    parser.add_argument(
      '-cpu',
      type=bool,
      default=False
    )
    
    unparsed = parser.parse_known_args()
    
    if unparsed[0].cpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    main(unparsed[0])