from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data


tf.set_random_seed(777)


def conv2d(x, w, stride_=1, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, stride_, stride_, 1], padding=padding)


def max_pool2d(x, kernel_=2, stride_=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernel_, kernel_, 1], strides=[1, stride_, stride_, 1], padding=padding)


def SELU(x):
    # fixed point mean, var (0, 1)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def LeakyReLU(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)

    return f1 * x + f2 * abs(x)


def MLPwithSELU(x, w, b, p_keep_conv, p_keep_hidden):
    with tf.variable_scope("SELU", reuse=None):
        net = tf.nn.bias_add(SELU(conv2d(x, w['w0'])), b['b0'])
        net = max_pool2d(net)
        net = tf.nn.dropout(net, p_keep_conv)

        net = tf.nn.bias_add(SELU(conv2d(net, w['w1'])), b['b1'])
        net = tf.nn.dropout(net, p_keep_conv - 0.05)

        net = tf.nn.bias_add(SELU(conv2d(net, w['w2'], 2)), b['b2'])
        net = tf.reshape(net, [-1, w['w3'].get_shape().as_list()[0]])
        net = tf.nn.dropout(net, p_keep_conv - 0.10)

        net = tf.add(SELU(tf.matmul(net, w['w3'])), b['b3'])
        net = tf.nn.dropout(net, p_keep_hidden)

        net = tf.add(tf.matmul(net, w['w4']), b['b4'])
        net = tf.contrib.layers.flatten(net)
    return net


def MLPwithELU(x, w, b, p_keep_conv, p_keep_hidden):
    with tf.variable_scope("ELU", reuse=None):
        net = tf.nn.bias_add(tf.nn.elu(conv2d(x, w['w0'])), b['b0'])
        net = max_pool2d(net)
        net = tf.nn.dropout(net, p_keep_conv)

        net = tf.nn.bias_add(tf.nn.elu(conv2d(net, w['w1'])), b['b1'])
        net = tf.nn.dropout(net, p_keep_conv - 0.05)

        net = tf.nn.bias_add(tf.nn.elu(conv2d(net, w['w2'], 2)), b['b2'])
        net = tf.reshape(net, [-1, w['w3'].get_shape().as_list()[0]])
        net = tf.nn.dropout(net, p_keep_conv - 0.10)

        net = tf.add(tf.nn.elu(tf.matmul(net, w['w3'])), b['b3'])
        net = tf.nn.dropout(net, p_keep_hidden)

        net = tf.add(tf.matmul(net, w['w4']), b['b4'])
        net = tf.contrib.layers.flatten(net)
    return net


def MLPwithLeakyReLU(x, w, b, p_keep_conv, p_keep_hidden):
    with tf.variable_scope("ReLU", reuse=None):
        net = tf.nn.bias_add(LeakyReLU(conv2d(x, w['w0'])), b['b0'])
        net = max_pool2d(net)
        net = tf.nn.dropout(net, p_keep_conv)

        net = tf.nn.bias_add(LeakyReLU(conv2d(net, w['w1'])), b['b1'])
        net = tf.nn.dropout(net, p_keep_conv - 0.05)

        net = tf.nn.bias_add(LeakyReLU(conv2d(net, w['w2'], 2)), b['b2'])
        net = tf.reshape(net, [-1, w['w3'].get_shape().as_list()[0]])
        net = tf.nn.dropout(net, p_keep_conv - 0.10)

        net = tf.add(LeakyReLU(tf.matmul(net, w['w3'])), b['b3'])
        net = tf.nn.dropout(net, p_keep_hidden)

        net = tf.add(tf.matmul(net, w['w4']), b['b4'])
        net = tf.contrib.layers.flatten(net)
    return net


def MLPwithReLU(x, w, b, p_keep_conv, p_keep_hidden):
    with tf.variable_scope("ReLU", reuse=None):
        net = tf.nn.bias_add(tf.nn.relu(conv2d(x, w['w0'])), b['b0'])
        net = max_pool2d(net)
        net = tf.nn.dropout(net, p_keep_conv)

        net = tf.nn.bias_add(tf.nn.relu(conv2d(net, w['w1'])), b['b1'])
        net = tf.nn.dropout(net, p_keep_conv - 0.05)

        net = tf.nn.bias_add(tf.nn.relu(conv2d(net, w['w2'], 2)), b['b2'])
        net = tf.reshape(net, [-1, w['w3'].get_shape().as_list()[0]])
        net = tf.nn.dropout(net, p_keep_conv - 0.10)

        net = tf.add(tf.nn.relu(tf.matmul(net, w['w3'])), b['b3'])
        net = tf.nn.dropout(net, p_keep_hidden)

        net = tf.add(tf.matmul(net, w['w4']), b['b4'])
        net = tf.contrib.layers.flatten(net)
    return net


start_time = time.time()  # clocking start

# load mnist dataset
mnist = input_data.read_data_sets('./MNIST', one_hot=True)

# placeholder for image, label
X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x-image")
Y = tf.placeholder(tf.float32, shape=[None, 10], name="y-label")

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# Weights & Biases
W = {
    'w0': tf.get_variable('w0', shape=[5, 5, 1, 32],
                          regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                          initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w1': tf.get_variable('w1', shape=[5, 5, 32, 64],
                          regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                          initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w2': tf.get_variable('w2', shape=[5, 5, 64, 128],
                          regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                          initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w3': tf.get_variable('w3', shape=[7 * 7 * 128, 512],
                          regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                          initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w4': tf.get_variable('w4', shape=[512, 10],
                          regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                          initializer=tf.contrib.layers.variance_scaling_initializer()),
}

# bias
b = {
    'b0': tf.Variable(tf.zeros([32])),
    'b1': tf.Variable(tf.zeros([64])),
    'b2': tf.Variable(tf.zeros([128])),
    'b3': tf.Variable(tf.zeros([512])),
    'b4': tf.Variable(tf.zeros([10]))
}

i = 1  # change this value

if i == 1:
    X_ = MLPwithSELU(X, W, b, p_keep_conv, p_keep_hidden)
elif i == 2:
    X_ = MLPwithReLU(X, W, b, p_keep_conv, p_keep_hidden)
elif i == 3:
    X_ = MLPwithELU(X, W, b, p_keep_conv, p_keep_hidden)
elif i == 4:
    X_ = MLPwithLeakyReLU(X, W, b, p_keep_conv, p_keep_hidden)

epochs = 25
batch_size = 64
batch = tf.Variable(0)
with tf.name_scope("cost/train/acc"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=X_))

    learning_rate = tf.train.exponential_decay(
        learning_rate=1e-3,
        global_step=batch * batch_size,
        decay_steps=mnist.train.num_examples,
        decay_rate=0.9,
        staircase=True
    )
    train = tf.train.AdamOptimizer(1e-3).minimize(cost, global_step=batch)

    prediction = tf.equal(tf.argmax(X_, 1), tf.argmax(Y, 1))
    accuaracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session() as s:
    s.run(tf.global_variables_initializer())

    max_acc = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    for epoch in range(1, epochs + 1):
        avg_cost = 0.
        for step in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape(-1, 28, 28, 1)

            _, train_acc = s.run([train, accuaracy], feed_dict={X: batch_x,
                                                                Y: batch_y,
                                                                p_keep_conv: 0.9,
                                                                p_keep_hidden: 0.5})
            avg_cost += s.run(cost, feed_dict={X: batch_x,
                                               Y: batch_y,
                                               p_keep_conv: 0.9,
                                               p_keep_hidden: 0.5}) / total_batch

            if step % 100 == 0:
                valid_acc = s.run(accuaracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                                        Y: mnist.test.labels,
                                                        p_keep_conv: 1.,
                                                        p_keep_hidden: 1.})
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    print("[+] Updated New max accuracy {:.4f}".format(max_acc))
                    saver.save(s, "./model/" + str(i) + '/', global_step=(step + epoch * mnist.train.num_examples))

        if epoch % 2 == 0:
            print("[*] Epoch %03d =>" % epoch, " Average Cost : {:.8f}".format(avg_cost))

    end_time = time.time() - start_time

    # elapsed time
    print("[+] Elapsed time {:.10f}s".format(end_time))

    s.close()
