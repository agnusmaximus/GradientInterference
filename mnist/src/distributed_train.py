from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from threading import Timer
import os.path
import time

import numpy as np
import random
import tensorflow as tf
import signal
import sys
import os
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
import mnist

np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

# Training defines
tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist_train', 'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_boolean('should_evaluate', False, 'Whether Chief should do evaluation per epoch.')
tf.app.flags.DEFINE_integer('evaluate_batch_size', 1000, 'Batchsize for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batchsize for training')
tf.app.flags.DEFINE_integer('checkpoint_save_secs', 60*5, 'Time interval between checkpoint saving')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Constant learning rate.')

# Distributed defines
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")


def train(target, dataset, cluster_spec):
  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])
  num_replicas_to_aggregate = num_workers
  assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                         'num_parameter_servers'
                                                         ' must be > 0.')
  is_chief = (FLAGS.task_id == 0)

  # Ops are assigned to worker by default.
  with tf.device(
      tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % FLAGS.task_id,
        cluster=cluster_spec)):

    # Create a variable to count the number of train() calls. This equals the
    # number of updates applied to the variables. The PS holds the global step.
    global_step = tf.Variable(0, name="global_step", trainable=False)


    # Decay the learning rate exponentially based on the number of steps.
    images, labels = mnist.placeholder_inputs(FLAGS.batch_size)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    logits = mnist.inference(images, train=True)

    # Validation accuracy operation
    val_acc = tf.reduce_sum(mnist.evaluation(logits, labels)) / tf.constant(FLAGS.evaluate_batch_size)

    # Add classification loss.
    total_loss = mnist.loss(logits, labels)

    # Create an optimizer that performs gradient descent.
    lr = tf.constant(FLAGS.learning_rate, tf.float32)
    opt = tf.train.GradientDescentOptimizer(lr)

    # Use V2 optimizer
    opt = tf.train.SyncReplicasOptimizer(
      opt,
      replicas_to_aggregate=num_replicas_to_aggregate,
      total_num_replicas=num_workers)

    # Compute gradients with respect to the loss.
    grads = opt.compute_gradients(total_loss)
    apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradients_op]):
      train_op = tf.identity(total_loss, name='train_op')

  sync_replicas_hook = opt.make_session_run_hook(is_chief)

  # We create queues to block and unblock workers for waiting on master to evaluate on training set
  with ops.device(global_step.device):
    block_workers_queue = data_flow_ops.FIFOQueue(1,
                                                  tf.int64,
                                                  shapes=(),
                                                  name="block_workers_queue",
                                                  shared_name="block_workers_queue")

    block_workers_op = block_workers_queue.enqueue(tf.constant(0, dtype=tf.int64))
    unblock_workers_op = block_workers_queue.dequeue()
    workers_block_if_necessary_op = tf.while_loop(lambda x : block_workers_queue.size() > 0,
                                                  lambda x : tf.constant(0),
                                                  [tf.constant(0)])

  # Helper function to evaluate on training set
  def model_evaluate(sess):
    tf.logging.info("Evaluating model...")
    sys.stdout.flush()

    num_examples = dataset.num_examples

    # This simply makes sure that we are evaluating on the training set
    assert(num_examples == 60000)

    # Make sure we are using a batchsize a multiple of number of examples
    assert(num_examples % FLAGS.evaluate_batch_size == 0)
    num_iter = int(num_examples / FLAGS.evaluate_batch_size)
    acc, loss = 0, 0

    for i in range(num_iter):
      feed_dict = mnist.fill_feed_dict(dataset, images, labels, batch_size)
      acc_p, loss_p = sess.run(
        [val_acc, val_loss], feed_dict=feed_dict)

      tf.logging.info("%d of %d" % (step, num_iter))
      sys.stdout.flush()

      acc += acc_p * batch_size
      loss += loss_p

    tf.logging.info("Done evaluating...")

    # Compute precision @ 1.
    acc /= float(num_examples)
    return acc, loss


  evaluate_times = []
  n_examples_processed = 0

  # Cur epoch track describes (from this worker's perspective) the current epoch.
  cur_epoch_track = 0
  first_iteration = True

  with tf.train.MonitoredTrainingSession(
      master=target, is_chief=is_chief,
      hooks=[sync_replicas_hook],
      checkpoint_dir=FLAGS.train_dir,
      save_checkpoint_secs=FLAGS.checkpoint_save_secs) as mon_sess:
    while not mon_sess.should_stop():

      # Compute current epoch and cast to integer
      new_epoch_float = n_examples_processed / float(dataset.num_examples)
      new_epoch_track = int(new_epoch_float)

      # For distributed training, we want to block other workers from continuing if the master is evaluating.
      mon_sess.run([workers_block_if_necessary_op])

      # On each epoch (or the first), evaluate on training data
      if FLAGS.should_evaluate and FLAGS.task_id == 0 and (new_epoch_track == cur_epoch_track+1 or first_iteration):

        # Block workers
        mon_sess.run([block_workers_op])

        # Keep track of elapsed time and discount it from the data
        t_evaluate_start = time.time()

        # Evaluate
        tf.logging.info("Master evaluating...")
        sys.stdout.flush()
        acc, loss = model_evaluate(mon_sess)
        tf.logging.info("IInfo: %f %f %f %f" % (t_evaluate_start-sum(evaluate_times), new_epoch_float, acc, loss))

        # Keep track of elapsed time
        t_evaluate_end = time.time()

        tf.logging.info("Master done evaluating... Elapsed time: %f" % (t_evaluate_end-t_evaluate_start))

        # Keep track of elapsed time for evaluation
        evaluate_times.append(t_evaluate_end-t_evaluate_start)

        # Unblock workers
        mon_sess.run([unblock_workers_op])

      # Update current epoch
      cur_epoch_track = max(cur_epoch_track, new_epoch_track)

      tf.logging.info("Epoch: %d" % int(cur_epoch_track))

      # Distributed training
      feed_dict = mnist.fill_feed_dict(dataset, images, labels, FLAGS.batch_size)
      loss_value  = mon_sess.run([train_op], feed_dict=feed_dict)
      n_examples_processed += FLAGS.batch_size * num_workers

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      first_iteration = False
