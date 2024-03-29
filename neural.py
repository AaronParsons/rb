'''This file defines the tensorflow neural network that is used for image
identification in the find module.  It also defines the NN that is trained
in the train module.'''
import tensorflow as tf; tf.logging.set_verbosity(tf.logging.ERROR)

# Define NN

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# VERSION 002
HALF_SZ = 16
IMG_SIZE = HALF_SZ * 2
NCHAN = 1
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, NCHAN])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# The convolution will compute 32 features for each 5x5 patch. 
# Its weight tensor will have a shape of [5, 5, 1, 32]. 
# The first two dimensions are the patch size, the next is the number of 
# input channels, and the last is the number of output channels. 
W_conv1 = weight_variable([5, 5, NCHAN, 32])
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 4d tensor, 
# with the second and third dimensions corresponding to image width and height, 
# and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, NCHAN])

# We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, 
# and finally max pool. The max_pool_2x2 method will reduce the image size to IMG_SIZE/2
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# In order to build a deep network, we stack several layers of this type. 
# The second layer will have 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Now that the image size has been reduced to 7x7, we add a fully-connected 
# layer with 1024 neurons to allow processing on the entire image. We reshape 
# the tensor from the pooling layer into a batch of vectors, multiply 
# by a weight matrix, add a bias, and apply a ReLU.
#W_fc1 = weight_variable([16 * 16 * 64, 128])
W_fc1 = weight_variable([16 * 16 * 16, 128])
b_fc1 = bias_variable([128])

#h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# To reduce overfitting, we will apply dropout before the readout layer. We
# create a placeholder for the probability that a neuron's output is kept
# during dropout. This allows us to turn dropout on during training, and turn
# it off during testing. TensorFlow's tf.nn.dropout op automatically handles
# scaling neuron outputs in addition to masking them, so dropout just works
# without any additional scaling.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 2])
b_fc2 = bias_variable([2])

h_fc = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = h_fc

# Training interface
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

