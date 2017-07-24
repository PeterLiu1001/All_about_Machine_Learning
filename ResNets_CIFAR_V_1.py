#!/disk/scratch/mlp/miniconda2/bin/python
import os
import datetime
import numpy as np
import tensorflow as tf
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
'''VGG_10'''


# check necessary environment variables are defined
assert 'MLP_DATA_DIR' in os.environ, (
    'An environment variable MLP_DATA_DIR must be set to the path containing'
    ' MLP data before running script.')
assert 'OUTPUT_DIR' in os.environ, (
    'An environment variable OUTPUT_DIR must be set to the path to write'
    ' output to before running script.')

'''In this section, I will load the data and reshape them for RGB channels'''

train_data_100 = CIFAR100DataProvider('train', batch_size=128)
valid_data_100 = CIFAR100DataProvider('valid', batch_size=128)

'''Reshape the train and valid data to -1X32X32X3'''
train_data_100.inputs = train_data_100.inputs.reshape((40000, -1, 3), order='F')
train_data_100.inputs = train_data_100.inputs.reshape((40000, 32, 32, 3))
valid_data_100.inputs = valid_data_100.inputs.reshape((10000, -1, 3), order='F')
valid_data_100.inputs = valid_data_100.inputs.reshape((10000, 32, 32, 3))

#change the valid targets to one hot coding
valid_targets = valid_data_100.to_one_of_k(valid_data_100.targets)

# Prepare some function for the net
def conv2d_stride1(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def conv2d_stride2(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def weight_variable_normal(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.zeros(shape=shape)
  return tf.Variable(initial)



#Adding data feeder for computation graph
with tf.name_scope('data_feeder'):
    inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], 'inputs')
    targets = tf.placeholder(tf.float32, [None, train_data_100.num_classes], 'targets')

#Adding the first convolutional layer into the graph
with tf.name_scope('Convolutional_layer_1'):
    W_conv1 = weight_variable_normal([3, 3, 3, 32])
    b_conv1 = bias_variable([32])
    conv_1 = tf.nn.relu(conv2d_stride1(inputs, W_conv1) + b_conv1)
with tf.name_scope('Convolutional_layer_2'):
    W_conv2 = weight_variable_normal([3, 3, 32, 32])
    b_conv2 = bias_variable([32])
    conv_2 = tf.nn.relu(conv2d_stride1(conv_1, W_conv2) + b_conv2)
with tf.name_scope('Max_pooling_2x2_layer_1'):
    pool1 = max_pool_2x2(conv_2)
with tf.name_scope('Local_response_normal_layer_1'):
    lrn1 = tf.nn.local_response_normalization(pool1)


with tf.name_scope('Convolutional_layer_3'):
    W_conv3 = weight_variable_normal([3, 3, 32, 64])
    b_conv3 = bias_variable([64])
    conv_3 = tf.nn.relu(conv2d_stride1(lrn1, W_conv3) + b_conv3)
with tf.name_scope('Convolutional_layer_4'):
    W_conv4 = weight_variable_normal([3, 3, 64, 64])
    b_conv4 = bias_variable([64])
    conv_4 = tf.nn.relu(conv2d_stride1(conv_3, W_conv4) + b_conv4)
with tf.name_scope('Max_pooling_2x2_layer_2'):
    pool2 = max_pool_2x2(conv_4)
with tf.name_scope('Local_response_normal_layer_2'):
    lrn2 = tf.nn.local_response_normalization(pool2)


with tf.name_scope('Convolutional_layer_5'):
    W_conv5 = weight_variable_normal([3, 3, 64, 128])
    b_conv5 = bias_variable([128])
    conv_5 = tf.nn.relu(conv2d_stride1(lrn2, W_conv5) + b_conv5)
with tf.name_scope('Convolutional_layer_6'):
    W_conv6 = weight_variable_normal([3, 3, 128, 128])
    b_conv6 = bias_variable([128])
    conv_6 = tf.nn.relu(conv2d_stride1(conv_5, W_conv6) + b_conv6)
with tf.name_scope('Max_pooling_2x2_layer_3'):
    pool3 = max_pool_2x2(conv_6)
with tf.name_scope('Local_response_normal_layer_3'):
    lrn3 = tf.nn.local_response_normalization(pool3)


with tf.name_scope('Convolutional_layer_7'):
    W_conv7 = weight_variable_normal([3, 3, 128, 256])
    b_conv7 = bias_variable([256])
    conv_7 = tf.nn.relu(conv2d_stride1(lrn3, W_conv7) + b_conv7)
with tf.name_scope('Convolutional_layer_8'):
    W_conv8 = weight_variable_normal([3, 3, 256, 256])
    b_conv8 = bias_variable([256])
    conv_8 = tf.nn.relu(conv2d_stride1(conv_7, W_conv8) + b_conv8)
with tf.name_scope('Max_pooling_2x2_layer_4'):
    pool4 = max_pool_2x2(conv_8)
with tf.name_scope('Local_response_normal_layer_4'):
    lrn4 = tf.nn.local_response_normalization(pool4)




#Adding the first fully connected layer into the graph
with tf.name_scope('Fully_connected_layer_1'):
    W_fc1 = weight_variable_normal([2*2*256, 1024])
    b_fc1 = bias_variable([1024])
    lrn4_flat = tf.reshape(lrn4, [-1, 2*2*256])
    fc_1 = tf.nn.relu(tf.matmul(lrn4_flat, W_fc1) + b_fc1)

#Adding the first dropout layer for reducing overfitting
with tf.name_scope('Drop_out_layer_1'):
    keep_prop = tf.placeholder(tf.float32)
    fc_1_drop_1 = tf.nn.dropout(fc_1, keep_prop)

#Adding the read out layer for computing softmax error
with tf.name_scope('Read_out_layer'):
    W_out = weight_variable_normal([1024, 100])
    b_out = bias_variable([100])
    ro = tf.matmul(fc_1_drop_1, W_out) + b_out

#Adding subgraph to compute the error
with tf.name_scope('error'):
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ro, targets))

#Adding subgraph to train the model
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(error)

#Adding subgraph to compute the accuracy per batch
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ro, 1), tf.argmax(targets, 1)), tf.float32))
#Adding subgraph to handle the log info produced during training process
with tf.name_scope('summary'):
    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

'''In this section, we will save the log info and checkpoints into the disk'''
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join(os.environ['OUTPUT_DIR'], timestamp)
checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
train_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train-summaries'))
valid_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'valid-summaries'))
saver = tf.train.Saver()

'''In this section, we will initialize variables used in this computation graph '''
num_epoch = 50
train_accuracy = np.zeros(num_epoch)
train_error = np.zeros(num_epoch)
valid_accuracy = np.zeros(num_epoch)
valid_error = np.zeros(num_epoch)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
step = 0


'''Now, train the model'''
for e in range(num_epoch):
    for b, (input_batch, target_batch) in enumerate(train_data_100):
        # do train step with current batch
        _ = sess.run(train_step, feed_dict={inputs: input_batch, targets: target_batch, keep_prop: 0.5})
        summary, batch_error, batch_acc = sess.run(
            [summary_op, error, accuracy],
            feed_dict={inputs: input_batch, targets: target_batch, keep_prop: 1.0})
        # add summary and accumulate stats
        train_writer.add_summary(summary, step)
        train_error[e] += batch_error
        train_accuracy[e] += batch_acc
        step += 1
    # normalise running means by number of batches
    train_error[e] /= train_data_100.num_batches
    train_accuracy[e] /= train_data_100.num_batches
    # evaluate validation set performance
    valid_summary, valid_error[e], valid_accuracy[e] = sess.run(
        [summary_op, error, accuracy],
        feed_dict={inputs: valid_data_100.inputs, targets: valid_targets, keep_prop: 1.0})
    valid_writer.add_summary(valid_summary, step)
    # checkpoint model variables
    saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), step)
    # write stats summary to stdout
    print('Epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
          .format(e + 1, train_error[e], train_accuracy[e]))
    print('          err(valid)={0:.2f} acc(valid)={1:.2f}'
          .format(valid_error[e], valid_accuracy[e]))

'''Close the sess and file writer when all the episode have been run'''
train_writer.close()
valid_writer.close()
sess.close()

'''Save the log info in disk'''
np.savez_compressed(
    os.path.join(exp_dir, 'run.npz'),
    train_error=train_error,
    train_accuracy=train_accuracy,
    valid_error=valid_error,
    valid_accuracy=valid_accuracy
)



