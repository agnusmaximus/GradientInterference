# This code is taken and modified from the inception_distribute_train.py file of
# google's tensorflow inception model. The original source is here - https://github.com/tensorflow/models/tree/master/inception.

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
import mnist_data

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops

from dcgan_model import DCGAN

np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('should_evaluate', False, 'Whether Chief should do evaluation per epoch.')
tf.app.flags.DEFINE_boolean('should_compute_R', False, 'Whether Chief should do compute R per epoch.')
tf.app.flags.DEFINE_integer('evaluate_batchsize', 1000,
                           """Batchsize for evaluation""")

tf.app.flags.DEFINE_boolean('should_summarize', False, 'Whether Chief should write summaries.')
tf.app.flags.DEFINE_boolean('timeline_logging', False, 'Whether to log timeline of events.')
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('rpc_port', 1235,
                           """Port for timeout communication""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 20,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 300,
                            'Save summaries interval seconds.')

#####################################
# The following flags are for DCGAN #
# Note that the DCGAN train method  #
# is never actually called.         #
#####################################
tf.app.flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
tf.app.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
tf.app.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
tf.app.flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
tf.app.flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
tf.app.flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
tf.app.flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
tf.app.flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
tf.app.flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
tf.app.flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
tf.app.flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
tf.app.flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
tf.app.flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
tf.app.flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
tf.app.flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
tf.app.flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

##############################

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
#tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
#                          'Initial learning rate.')
# For flowers
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.999,
                          'Learning rate decay factor.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def model_evaluate(sess, dataset, dcgan, batch_size, d_loss, g_loss):
  tf.logging.info("Evaluating model...")
  n_iters = int(math.ceil(dataset.num_examples / batch_size))
  assert(dataset.num_examples % batch_size == 0)
  d_loss_total, g_loss_total = 0, 0
  tf.logging.info("Starting")
  sys.stdout.flush()
  for i in range(n_iters):
    tf.logging.info("%d of %d" % (i, n_iters))
    sys.stdout.flush()
    images_real, labels_real = dataset.next_batch(FLAGS.batch_size)
    tf.logging.info("%d of %d" % (i, n_iters))
    sys.stdout.flush()
    batch_z = np.random.uniform(-1, 1, [batch_size, dcgan.z_dim]).astype(np.float32)
    tf.logging.info("%d of %d" % (i, n_iters))
    sys.stdout.flush()
    feed_dict = {
      dcgan.z : batch_z,
      dcgan.y : labels_real,
      dcgan.inputs : images_real,
    }
    d_loss, g_loss = sess.run([d_loss, g_loss], fd=feed_dict)
    tf.logging.info("%d of %d" % (i, n_iters))
    sys.stdout.flush()

    d_loss_total += d_loss
    g_loss_total += g_loss
    tf.logging.info("%d of %d" % (i, n_iters))
    sys.stdout.flush()

  return d_loss_total, g_loss_total

def train(target, dataset, cluster_spec):

  """Train Inception on a dataset for a number of steps."""
  # Number of workers and parameter servers are infered from the workers and ps
  # hosts string.
  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])
  # If no value is given, num_replicas_to_aggregate defaults to be the number of
  # workers.
  if FLAGS.num_replicas_to_aggregate == -1:
    num_replicas_to_aggregate = num_workers
  else:
    num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

  # Both should be greater than 0 in a distributed training.
  assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                         'num_parameter_servers'
                                                         ' must be > 0.')

  # Choose worker 0 as the chief. Note that any worker could be the chief
  # but there should be only one chief.
  is_chief = (FLAGS.task_id == 0)

  # Ops are assigned to worker by default.
  with tf.device(
      tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % FLAGS.task_id,
        cluster=cluster_spec)):

    # Create a variable to count the number of train() calls. This equals the
    # number of updates applied to the variables. The PS holds the global step.
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Create the dcgan
    # The first argument is session, which is only used for training.
    # We are not using their training method, so we pass None.
    #
    # Also note that some variables like checkpoint_dir are also used
    # only for their training routine. Since their training routine is
    # never called, these variables are never used (like FLAGS.checkpoint_dir).
    dcgan = DCGAN(
        None,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        y_dim=10,
        c_dim=1,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        is_crop=FLAGS.is_crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)

    d_loss, d_vars = dcgan.d_loss, dcgan.d_vars
    g_loss, g_vars = dcgan.g_loss, dcgan.g_vars

    tf.logging.info("Discriminator variables %s" % str(list([str(x) for x in dcgan.d_vars])))
    tf.logging.info("Generator variables %s" % str(list([str(x) for x in dcgan.g_vars])))

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate)

    # Use V2 optimizer
    opt = tf.train.SyncReplicasOptimizer(
      opt,
      replicas_to_aggregate=num_replicas_to_aggregate,
      total_num_replicas=num_workers)

    # Compute gradients with respect to the loss.
    grads_d, grads_g = opt.compute_gradients(d_loss), opt.compute_gradients(g_loss)
    apply_gradients_g = opt.apply_gradients(grads_d, global_step=global_step)
    apply_gradients_d = opt.apply_gradients(grads_d, global_step=global_step)

    with tf.control_dependencies([apply_gradients_g]):
      train_op_g = tf.identity(g_loss, name='train_op_g')

    with tf.control_dependencies([apply_gradients_d]):
      train_op_d = tf.identity(d_loss, name='train_op_d')

    # Queue for broadcasting R
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

    work_image_placeholder = tf.placeholder(tf.float32, shape=(1, FLAGS.input_width, FLAGS.input_height, FLAGS.c_dim))
    work_label_placeholder = tf.placeholder(tf.int64, shape=(None,))

    # Queue for distributing computation of R
    with ops.device(global_step.device):
      R_images_work_queue = []
      R_labels_work_queue = []
      for i in range(num_workers):
        name_images = "r_images_work_queue_%d" % i
        name_labels = "r_labels_work_queue_%d" % i
        R_images_work_queue.append(data_flow_ops.FIFOQueue(-1, tf.float32, name=name_images, shared_name=name_images))
        R_labels_work_queue.append(data_flow_ops.FIFOQueue(-1, tf.int64, name=name_labels, shared_name=name_labels))

      gradient_sums_queue = data_flow_ops.FIFOQueue(-1, tf.float32, name="gradient_sums_queue", shared_name="gradient_sums_queue")
      sum_of_norms_queue = data_flow_ops.FIFOQueue(-1, tf.float32, name="gradient_norms_queue", shared_name="gradient_norms_queue")

      R_computed_step = tf.Variable(0, name="R_computed_step", trainable=False)

    gradient_sum_placeholder = tf.placeholder(tf.float32, shape=(None))
    gradient_sums_enqueue = gradient_sums_queue.enqueue(gradient_sum_placeholder)
    gradient_sums_dequeue = gradient_sums_queue.dequeue()

    sum_of_norms_placeholder = tf.placeholder(tf.float32, shape=())
    sum_of_norms_enqueue = sum_of_norms_queue.enqueue(sum_of_norms_placeholder)
    sum_of_norms_dequeue = sum_of_norms_queue.dequeue()

    gradients_sums_size = gradient_sums_queue.size()
    sum_of_norms_size = sum_of_norms_queue.size()

    step_placeholder = tf.placeholder(tf.int32, shape=(None))
    update_r_computed_step = R_computed_step.assign(step_placeholder)

    # Enqueue operations for adding work to the R queue
    enqueue_image_ops_for_r = []
    enqueue_label_ops_for_r = []
    for i in range(num_workers):
      enqueue_image_ops_for_r.append(R_images_work_queue[i].enqueue(work_image_placeholder))
      enqueue_label_ops_for_r.append(R_labels_work_queue[i].enqueue(work_label_placeholder))

    length_of_images_queue = []
    length_of_labels_queue = []
    dequeue_work_images = []
    dequeue_label_images = []
    for i in range(num_workers):
      length_of_images_queue.append(R_images_work_queue[i].size())
      length_of_labels_queue.append(R_labels_work_queue[i].size())
      dequeue_work_images.append(R_images_work_queue[i].dequeue())
      dequeue_label_images.append(R_labels_work_queue[i].dequeue())

  def distributed_compute_R(sess, cur_step):
    # Incomplete.
    return 0

  sync_replicas_hook = opt.make_session_run_hook(is_chief)

  # specified interval. Note that the summary_op and train_op never run
  # simultaneously in order to prevent running out of GPU memory.
  next_summary_time = time.time() + FLAGS.save_summaries_secs
  begin_time = time.time()
  cur_iteration = -1
  iterations_finished = set()

  n_examples_processed = 0
  cur_epoch_track = 0
  compute_R_train_error_time = 0
  loss_value = -1
  step = -1

  checkpoint_save_secs = 60*5

  compute_R_times, evaluate_times = [0], [0]

  tf.logging.info("Starting training session...")

  with tf.train.MonitoredTrainingSession(
      master=target, is_chief=is_chief,
      hooks=[sync_replicas_hook],
      checkpoint_dir=FLAGS.train_dir,
      save_checkpoint_secs=checkpoint_save_secs) as mon_sess:
    while not mon_sess.should_stop():

      default_batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size, dcgan.z_dim]).astype(np.float32)
      default_images = np.zeros((FLAGS.batch_size, mnist_data.IMAGE_SIZE, mnist_data.IMAGE_SIZE, 1))
      default_labels = np.zeros((FLAGS.batch_size, mnist_data.NUM_LABELS))
      default_fd = {dcgan.z : default_batch_z, dcgan.y : default_labels, dcgan.inputs : default_images}

      cur_iteration += 1
      sys.stdout.flush()

      start_time = time.time()

      run_options = tf.RunOptions()
      run_metadata = tf.RunMetadata()

      if FLAGS.timeline_logging:
        run_options.trace_level=tf.RunOptions.FULL_TRACE
        run_options.output_partition_graphs=True

      # Compute batchsize ratio
      new_epoch_float = n_examples_processed / float(dataset.num_examples)
      new_epoch_track = int(new_epoch_float)

      # Block workers if necessary if master is computing R or evaluating
      mon_sess.run([workers_block_if_necessary_op], feed_dict=default_fd)

      if FLAGS.should_evaluate and FLAGS.task_id == 0 and (new_epoch_track == cur_epoch_track+1 or cur_iteration == 0):
        mon_sess.run([block_workers_op], feed_dict=default_fd)
        t_evaluate_start = time.time()
        tf.logging.info("Master evaluating...")
        d_loss_value, g_loss_value = model_evaluate(mon_sess, dataset, dcgan, FLAGS.evaluate_batchsize, d_loss, g_loss)
        tf.logging.info("IInfo: %f %f %f %f" % (t_evaluate_start-sum(evaluate_times)-sum(compute_R_times), new_epoch_float, d_loss_value, g_loss_value))
        t_evaluate_end = time.time()
        tf.logging.info("Master done evaluating... Elapsed time: %f" % (t_evaluate_end-t_evaluate_start))
        evaluate_times.append(t_evaluate_end-t_evaluate_start)
        mon_sess.run([unblock_workers_op], feed_dict=default_fd)

      num_steps_per_epoch = int(dataset.num_examples / (num_workers * FLAGS.batch_size))

      # We use step since it's synchronized across workers
      # Step != 0 is a hack to make sure R isn't computed twice in the beginning
      if (step % num_steps_per_epoch == 0 and step != 0) or step == -1:
        if FLAGS.should_compute_R and FLAGS.task_id == 0:
          t_compute_r_start = time.time()
          tf.logging.info("Master computing R...")
          R = distributed_compute_R(mon_sess, step)
          tf.logging.info("R: %f %f %f" % (t_compute_r_start-sum(evaluate_times)-sum(compute_R_times), R, new_epoch_float))
          t_compute_r_end = time.time()
          tf.logging.info("Master done computing R... Elapsed time: %f" % (t_compute_r_end-t_compute_r_start))
          compute_R_times.append(t_compute_r_end-t_compute_r_start)
        if FLAGS.should_compute_R and FLAGS.task_id != 0:
          distributed_compute_R(mon_sess, step)

      cur_epoch_track = max(cur_epoch_track, new_epoch_track)

      tf.logging.info("Epoch: %d" % int(cur_epoch_track))

      # Distributed training
      batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size, dcgan.z_dim]).astype(np.float32)

      # Train the discriminator
      images_real, labels_real = dataset.next_batch(FLAGS.batch_size)
      fd_d = {dcgan.inputs : images_real,
              dcgan.z : batch_z,
              dcgan.y : labels_real}
      loss_value_d, step_d = mon_sess.run([train_op_d, global_step], run_metadata=run_metadata, options=run_options, feed_dict=fd_d)

      # Train the generator
      fd_g = {dcgan.z : batch_z,
              dcgan.y : labels_real,
              dcgan.inputs : images_real}
      loss_value_g, step_g = mon_sess.run([train_op_g, global_step], run_metadata=run_metadata, options=run_options, feed_dict=fd_g)

      tf.logging.info("Step %d, d_loss: %f, g_loss: %f" % (step_g, loss_value_d, loss_value_g))

      n_examples_processed += FLAGS.batch_size * num_workers

      # This uses the queuerunner which does not support variable batch sizes
      #loss_value, step = sess.run([train_op, global_step], run_metadata=run_metadata, options=run_options)
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      # Log the elapsed time per iteration
      finish_time = time.time()

      # Create the Timeline object, and write it to a json
      if FLAGS.timeline_logging:
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('%s/worker=%d_timeline_iter=%d.json' % (FLAGS.train_dir, FLAGS.task_id, step), 'w') as f:
          f.write(ctf)

      if step > FLAGS.max_steps:
        break

      cur_epoch = n_examples_processed / float(dataset.num_examples)
      tf.logging.info("epoch: %f time %f" % (cur_epoch, time.time()-begin_time));

      #if cur_epoch >= FLAGS.n_train_epochs:
      #  break

      duration = time.time() - start_time
      examples_per_sec = FLAGS.batch_size / float(duration)
      format_str = ('Worker %d: %s: step %d, loss = %f'
                    '(%.1f examples/sec; %.3f  sec/batch)')
      tf.logging.info(format_str %
                      (FLAGS.task_id, datetime.now(), step, loss_value,
                       examples_per_sec, duration))

      # Determine if the summary_op should be run on the chief worker.
      if is_chief and next_summary_time < time.time() and FLAGS.should_summarize:

        tf.logging.info('Running Summary operation on the chief.')
        summary_str = mon_sess.run(summary_op)
        sv.summary_computed(sess, summary_str)
        tf.logging.info('Finished running Summary operation.')

        # Determine the next time for running the summary.
        next_summary_time += FLAGS.save_summaries_secs

  if is_chief:
    tf.logging.info('Elapsed Time: %f' % (time.time()-begin_time))

  # Stop the supervisor.  This also waits for service threads to finish.
  sv.stop()

  # Save after the training ends.
  if is_chief:
    saver.save(sess,
               os.path.join(FLAGS.train_dir, 'model.ckpt'),
               global_step=global_step)
