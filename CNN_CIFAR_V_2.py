#!/disk/scratch/mlp/miniconda2/bin/python
import os
import datetime
import numpy as np
import tensorflow as tf
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
'''
In version two, I will implement Alexnet on CIFAT dataset
'''


'''Version 1:
    1. The input data with the shape [batch, 32, 32, 3]
    2. The first convolutional layer: 5x5 feature map size, 32 channels, same padding, stride [1, 1, 1, 1]
    3. The first maxpooling (2x2 stride 2x2) layer: ksize [1, 2, 2, 1], stride [1, 2, 2, 1]
    4. Lrn layer
    5. The second convolutional layer: 5x5 feature map size, 64 channels, same padding, stride [1, 1, 1, 1]
    6. Lrn layer
    7. The second maxpooling (2x2 stride 2x2) layer: ksize [1, 2, 2, 1], stride [1, 2, 2, 1]
    8. Fully connected layer: 1024 hidden neurons [8*8*64, 1024]
       There are two 2x2 maxpooling layer so the size of the feature map is 8x8 with 64 channels (32/2/2)
       So, the parameters shape for this layer should be [8x8x64, #hidden neurons]. Need to reshape
       the output from conv layer 2 from 8x8x64 to 8*8*64 as the second dimension and the first dimension
       is the batch size 128 for this experiment. So, the input for this fully connected layer is
       [128, 8*8*64]
    9. Drop out layer for reducing overfitting: 0.5 as the drop probability. Note that, 0.5 drop
       probability should only used in training process and when computing the error and acc for log
       info, we should use 1.0 as the drop probability means drop nothing.
    10. Read out layer: 10 hidden neurons fully connected to the drop out layer so the shape of
        parameters in this layer is 1024x10.
'''


# check necessary environment variables are defined
assert 'MLP_DATA_DIR' in os.environ, (
    'An environment variable MLP_DATA_DIR must be set to the path containing'
    ' MLP data before running script.')
assert 'OUTPUT_DIR' in os.environ, (
    'An environment variable OUTPUT_DIR must be set to the path to write'
    ' output to before running script.')

'''In this section, I will load the data and reshape them for RGB channels'''
train_data_10 = CIFAR10DataProvider('train', batch_size=128)
valid_data_10 = CIFAR10DataProvider('valid', batch_size=128)
#train_data_100 = CIFAR100DataProvider('train', batch_size=50)
#valid_data_100 = CIFAR100DataProvider('valid', batch_size=50)

'''Reshape the train and valid data to -1X32X32X3'''
train_data_10.inputs = train_data_10.inputs.reshape((40000, -1, 3), order='F')
train_data_10.inputs = train_data_10.inputs.reshape((40000, 32, 32, 3))
valid_data_10.inputs = valid_data_10.inputs.reshape((10000, -1, 3), order='F')
valid_data_10.inputs = valid_data_10.inputs.reshape((10000, 32, 32, 3))

#change the valid targets to one hot coding
valid_targets = valid_data_10.to_one_of_k(valid_data_10.targets)

# Prepare some function for the net


def conv2d_stride1(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def conv2d_stride2(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Try different initialize methods for weight matrix
def weight_variable_normal_g(shape):
  fan_in = shape[0] * shape[1] * shape[2]
  fan_out = shape[0] * shape[1] * shape[3]
  initial = tf.truncated_normal(shape, stddev=np.sqrt(2. / (fan_in + fan_out)))
  return tf.Variable(initial)
def weight_variable_uniform_g(shape):
  fan_in = shape[0] * shape[1] * shape[2]
  fan_out = shape[0] * shape[1] * shape[3]
  initial = tf.random_uniform(shape, minval=-np.sqrt(6. / (fan_in + fan_out)), maxval=np.sqrt(6. / (fan_in + fan_out)))
  return tf.Variable(initial)
def weight_variable_normal_g_fc(shape):
  fan_in = shape[0]
  fan_out = shape[1]
  initial = tf.truncated_normal(shape, stddev=np.sqrt(2. / (fan_in + fan_out)))
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.zeros(shape=shape)
  return tf.Variable(initial)



#Adding data feeder for computation graph
with tf.name_scope('data_feeder'):
    inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], 'inputs')
    targets = tf.placeholder(tf.float32, [None, train_data_10.num_classes], 'targets')

#Adding the first convolutional layer into the graph
with tf.name_scope('Convolutional_layer_1'):
    W_conv1 = weight_variable_normal_g([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    conv_1 = tf.nn.elu(conv2d_stride1(inputs, W_conv1) + b_conv1)

#Adding the first 2x2 max pooling layer into the graph
with tf.name_scope('Max_pooling_2x2_layer_1'):
    pool1 = max_pool_2x2(conv_1)
#Adding the first local response normal layer into the graph
with tf.name_scope('Local_response_normal_layer_1'):
    lrn1 = tf.nn.local_response_normalization(pool1)

#Adding the second convolutional layer into the graph
with tf.name_scope('Convolutional_layer_2'):
    W_conv2 = weight_variable_normal_g([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    conv_2 = tf.nn.elu(conv2d_stride1(lrn1, W_conv2) + b_conv2)

#Adding the second local response normal layer into the graph
with tf.name_scope('Local_response_normal_layer_2'):
    lrn2 = tf.nn.local_response_normalization(conv_2)

#Adding the second 2x2 max pooling layer into the graph
with tf.name_scope('Max_pooling_2x2_layer_2'):
    pool2 = max_pool_2x2(lrn2)

#Adding the first fully connected layer into the graph
with tf.name_scope('Fully_connected_layer_1'):
    W_fc1 = weight_variable_normal_g_fc([8*8*64, 1024])
    b_fc1 = bias_variable([1024])
    pool2_flat = tf.reshape(pool2, [-1, 8*8*64])
    fc_1 = tf.nn.elu(tf.matmul(pool2_flat, W_fc1) + b_fc1)

#Adding the first dropout layer for reducing overfitting
with tf.name_scope('Drop_out_layer_1'):
    keep_prop = tf.placeholder(tf.float32)
    fc_1_drop_1 = tf.nn.dropout(fc_1, keep_prop)

#Adding the read out layer for computing softmax error
with tf.name_scope('Read_out_layer'):
    W_out = weight_variable_normal_g_fc([1024, 10])
    b_out = bias_variable([10])
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
    for b, (input_batch, target_batch) in enumerate(train_data_10):
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
    train_error[e] /= train_data_10.num_batches
    train_accuracy[e] /= train_data_10.num_batches
    # evaluate validation set performance
    valid_summary, valid_error[e], valid_accuracy[e] = sess.run(
        [summary_op, error, accuracy],
        feed_dict={inputs: valid_data_10.inputs, targets: valid_targets, keep_prop: 1.0})
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



