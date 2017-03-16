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

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import input as tf_input

import cifar_input
import resnet_model

IMAGE_SIZE = 32

np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('should_evaluate', False, 'Whether Chief should do evaluation per epoch.')
tf.app.flags.DEFINE_boolean('should_compute_R', False, 'Whether Chief should do compute R per epoch.')
tf.app.flags.DEFINE_integer('compute_r_batchsize', 100,
                           """Batchsize for computing r""")

tf.app.flags.DEFINE_boolean('n_train_epochs', 1000, 'Number of epochs to train for')
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

tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('rpc_port', 1235,
                           """Port for timeout communication""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('variable_batchsize', False,
                            'Use variable batchsize comptued using R.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 300,
                            'Save summaries interval seconds.')

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
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.999,
                          'Learning rate decay factor.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

EVAL_BATCHSIZE=2000

def compute_R_distributed(sess, model, inputs_dq_for_batchsize, images_pl, labels_pl, individual_gradients, batchsize):
  tf.logging.info("YAYAYAY")
  sys.stdout.flush()
  num_examples = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  tf.logging.info("YAYAYAY")
  sys.stdout.flush()
  num_iter = int(num_examples / batchsize)
  tf.logging.info("YAYAYAY")
  sys.stdout.flush()

  sum_of_norms, norm_of_sums = None, None
  tf.logging.info("YAYAYAY")
  sys.stdout.flush()

  for i in range(num_iter):
    tf.logging.info("computing r %d of %d" % (i, num_iter))
    sys.stdout.flush()

    t1 = time.time()
    images_real, labels_real = sess.run(inputs_dq_for_batchsize, feed_dict={images_pl:np.zeros([1, 32, 32, 3]), labels_pl: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})
    sys.stdout.flush()
    feed_dict = {images_pl:images_real, labels_pl:labels_real}
    sys.stdout.flush()
    gradients_real = sess.run(individual_gradients, feed_dict=feed_dict)
    t2= time.time()
    tf.logging.info("Time per iter: %f" % (t2-t1))
    sys.stdout.flush()

    assert(len(gradients_real) == batchsize)
    for gradients in gradients_real:
      gradient = np.concatenate(np.array([x.flatten() for x in gradients]))
      sys.stdout.flush()

      if sum_of_norms == None:
        sum_of_norms = np.linalg.norm(gradient)**2
      else:
        sum_of_norms += np.linalg.norm(gradient)**2

      if norm_of_sums == None:
        norm_of_sums = gradient
      else:
        norm_of_sums += gradient


  ratio_R = num_iter * FLAGS.batch_size * sum_of_norms / np.linalg.norm(norm_of_sums)**2
  return ratio_R

def model_evaluate(sess, model, images_pl, labels_pl, inputs_dq, batchsize):
  tf.logging.info("Evaluating model...")
  num_iter = int(math.ceil(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batchsize))
  correct_prediction, total_prediction = 0, 0
  total_sample_count = num_iter * batchsize
  computed_loss = 0
  step = 0

  while step < num_iter:
    images_real, labels_real = sess.run(inputs_dq, feed_dict={images_pl:np.zeros([1, 32, 32, 3]), labels_pl: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})
    feed_dict = {images_pl:images_real, labels_pl:labels_real}
    (summaries, loss, predictions, truth) = sess.run(
      [model.summaries, model.cost, model.predictions,
       model.labels], feed_dict=feed_dict)

    tf.logging.info("%d of %d" % (step, num_iter))

    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]
    computed_loss += loss
    step += 1

  tf.logging.info("Done evaluating...")

  # Compute precision @ 1.
  precision = 1.0 * correct_prediction / total_prediction
  return precision, computed_loss

def train(target, cluster_spec):

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

    global_step = tf.Variable(0, name="global_step", trainable=False)


    # Create a variable to count the number of train() calls. This equals the
    # number of updates applied to the variables. The PS holds the global step.

    #images, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.data_dir, FLAGS.batch_size, "train")
    images, labels = cifar_input.placeholder_inputs()
    variable_batchsize_inputs = cifar_input.build_input_multi_batchsize(FLAGS.dataset, FLAGS.data_dir, FLAGS.batch_size, "train")

    hps = resnet_model.HParams(batch_size=FLAGS.batch_size,
                               num_classes=10 if FLAGS.dataset=="cifar10" else 100,
                               min_lrn_rate=0.0001,
                               lrn_rate=FLAGS.initial_learning_rate,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='sgd')

    model = resnet_model.ResNet(hps, images, labels, "train")
    model.build_graph()

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate)

    opt = tf.train.SyncReplicasOptimizer(
      opt,
      replicas_to_aggregate=num_replicas_to_aggregate,
      total_num_replicas=num_workers,
    )

    # Compute gradients with respect to the loss.
    grads = opt.compute_gradients(model.cost)
    apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradients_op]):
        train_op = tf.identity(model.cost, name='train_op')

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

    work_image_placeholder = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, 3))
    work_label_placeholder = tf.placeholder(tf.int64, shape=(1, 10 if FLAGS.dataset == 'cifar10' else 100))

    # Queue for distributing computation of R
    with ops.device(global_step.device):
      R_images_work_queue = []
      R_labels_work_queue = []
      for i in range(num_workers):
        R_images_work_queue.append(data_flow_ops.FIFOQueue(-1, tf.float32, shapes=(work_image_placeholder.shape)))
        R_labels_work_queue.append(data_flow_ops.FIFOQueue(-1, tf.int64, shapes=(work_label_placeholder.shape)))

      gradient_sums_queue = data_flow_ops.FIFOQueue(-1, tf.float32)
      sum_of_norms_queue = data_flow_ops.FIFOQueue(-1, tf.float32)

    gradient_sum_placeholder = tf.placeholder(tf.float32, shape=(None))
    gradient_sums_enqueue = gradient_sums_queue.enqueue(gradient_sum_placeholder)

    sum_of_norms_placeholder = tf.placeholder(tf.float32, shape=())
    sum_of_norms_enqueue = sum_of_norms_queue.enqueue(sum_of_norms_placeholder)

    gradients_sums_size = gradient_sums_queue.size()
    sum_of_norms_size = sum_of_norms_queue.size()

    # Enqueue operations for adding work to the R queue
    enqueue_image_ops_for_r = []
    enqueue_label_ops_for_r = []
    for i in range(num_workers):
      #enqueue_image_ops_for_r.append(R_images_work_queue[i].enqueue(work_image_placeholder))
      #enqueue_label_ops_for_r.append(R_labels_work_queue[i].enqueue(work_label_placeholder))
      enqueue_label_ops_for_r.append(tf.Print(global_step, [global_step], message="testing"))
      enqueue_image_ops_for_r.append(tf.Print(global_step, [global_step], message="testing"))

    length_of_images_queue = []
    length_of_labels_queue = []
    dequeue_work_images = []
    dequeue_label_images = []
    for i in range(num_workers):
      length_of_images_queue.append(R_images_work_queue[i].size())
      length_of_labels_queue.append(R_labels_work_queue[i].size())
      dequeue_work_images.append(R_images_work_queue[i].dequeue())
      dequeue_label_images.append(R_labels_work_queue[i].dequeue())

  def distributed_compute_R(sess):

    worker_id = FLAGS.task_id

    # We block the work distribution so that when the workers pass this checkpoint,
    # all its work is in its queue.
    mon_sess.run([block_workers_op],feed_dict={images:np.zeros([1, 32, 32, 3]), labels: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})
    # Assign examples to workers
    if worker_id == 0:
      tf.logging.info("Master distributing examples for computing R...")
      for i in range(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN):
        img_work, label_work = sess.run(variable_batchsize_inputs[1], feed_dict={images:np.zeros([1, 32, 32, 3]), labels: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})
        tf.logging.info(str(img_work.shape) + " " + str(label_work.shape))
        worker = i % num_workers
        tf.logging.info("Assigning example %d to worker %d for computing R..." % (i, worker))
        #feed_dict={images:np.zeros([1, 32, 32, 3]), labels: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])}
        feed_dict={}
        feed_dict[work_image_placeholder] = img_work
        feed_dict[work_label_placeholder] = img_label
        #sess.run([enqueue_image_ops_for_r[i], enqueue_label_ops_for_r[i]], feed_dict=feed_dict)
        sess.run([enqueue_label_ops_for_r[i]])
      tf.logging.info("Master done distributing examples for computing R...")
    mon_sess.run([unblock_workers_op],feed_dict={images:np.zeros([1, 32, 32, 3]), labels: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})

    # For every worker, we pop from its queue and compute R on them
    n_labels_in_queue, n_images_in_queue = -1, -1
    sum_of_norms, norm_of_sums = None, None
    n_examples_computed = 0
    while n_labels_in_queue != 0:
      n_labels_in_queue, n_images_in_queue = sess.run([length_of_images_queue[worker_id],
                                                       length_of_labels_queue[worker_id]])
      assert(n_labels_in_queue == n_images_in_queue)
      if n_labels_in_queue == 0:
        break
      work_image, work_label = sess.run([dequeue_work_images[worker_id],
                                         dequeue_label_images[worker_id]])
      feed_dict = {images : work_image, label : work_label}
      gradients = sess.run(grad, feed_dict=feed_dict)
      gradient = np.concatenate(np.array([x.flatten() for x in gradients]))
      tf.logging.info("Worker computing r on examples...")

      if sum_of_norms == None:
        sum_of_norms = np.linalg.norm(gradient)**2
      else:
        sum_of_norms += np.linalg.norm(gradient)**2

      if norm_of_sums == None:
        norm_of_sums = gradient
      else:
        norm_of_sums += gradient

    # Worker has computed at least one example -- submit components of R
    if sum_of_norms != None:
      fd = {sum_of_norms_placeholder : sum_of_norms,
            gradient_sum_placeholder : norm_of_sums}

      tf.logging.info("Worker submitting sum of norms and norm of sums to queue...")
      sess.run([gradient_sums_enqueue, sum_of_norms_enqueue], feed_dict=fd)

    # Master waits until there are at least num_worker values in sum of gradients queue
    if worker_id == 0:
      tf.logging.info("Master waiting for num workers R components to be submitted...")
      n_gradient_sums, n_norm_sums = 0, 0
      while n_gradient_sums != n_workers and n_norm_sums != n_workers:
        n_gradient_sums, n_norm_sums = sess.run([gradients_sums_size, sum_of_norms_size])
        tf.logging.info("Accumulated %d gradient sums, %d norm sums (out of %d workers)" % (n_gradient_sums, n_norm_sums, num_workers))
      tf.logging.info("Master successfully received num workers components for R...")




  sync_replicas_hook = opt.make_session_run_hook(is_chief)

  # Train, checking for Nans. Concurrently run the summary operation at a
  # specified interval. Note that the summary_op and train_op never run
  # simultaneously in order to prevent running out of GPU memory.
  next_summary_time = time.time() + FLAGS.save_summaries_secs
  begin_time = time.time()

  # Keep track of own iteration
  cur_iteration = -1
  iterations_finished = set()

  n_examples_processed = 0
  cur_epoch_track = 0
  compute_R_train_error_time = 0
  loss_value = -1

  checkpoint_save_secs = 60*5

  compute_R_times, evaluate_times = [0], [0]

  with tf.train.MonitoredTrainingSession(
      master=target, is_chief=is_chief,
      hooks=[sync_replicas_hook],
      checkpoint_dir=FLAGS.train_dir,
      save_checkpoint_secs=checkpoint_save_secs) as mon_sess:
    while not mon_sess.should_stop():
      cur_iteration += 1
      sys.stdout.flush()

      start_time = time.time()

      run_options = tf.RunOptions()
      run_metadata = tf.RunMetadata()

      if FLAGS.timeline_logging:
        run_options.trace_level=tf.RunOptions.FULL_TRACE
        run_options.output_partition_graphs=True

      # Compute batchsize ratio
      new_epoch_float = n_examples_processed / float(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
      new_epoch_track = int(new_epoch_float)

      # Block workers if necessary if master is evaluating
      mon_sess.run([workers_block_if_necessary_op],feed_dict={images:np.zeros([1, 32, 32, 3]), labels: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})

      if FLAGS.should_evaluate and FLAGS.task_id == 0 and (new_epoch_track == cur_epoch_track+1 or cur_iteration == 0):
        mon_sess.run([block_workers_op],feed_dict={images:np.zeros([1, 32, 32, 3]), labels: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})
        t_evaluate_start = time.time()
        tf.logging.info("Master evaluating...")
        computed_precision, computed_loss = model_evaluate(mon_sess, model, images, labels, variable_batchsize_inputs[1000], 1000)
        t_evaluate_end = time.time()
        tf.logging.info("IInfo: %f %f %f %f" % (t_evaluate_start-sum(evaluate_times)-sum(compute_R_times), new_epoch_float, computed_precision, computed_loss))
        tf.logging.info("Master done evaluating... Elapsed time: %f" % (t_evaluate_end-t_evaluate_start))
        evaluate_times.append(t_evaluate_end-t_evaluate_start)
        mon_sess.run([unblock_workers_op],feed_dict={images:np.zeros([1, 32, 32, 3]), labels: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})

      if FLAGS.should_compute_R and FLAGS.task_id == 0 and (new_epoch_track == cur_epoch_track+1 or cur_iteration == 0):

        t_compute_r_start = time.time()
        tf.logging.info("Master computing R...")
        R = distributed_compute_R(mon_sess)
        tf.logging.info("R: %f %f" % (t_compute_r_start-sum(evaluate_times)-sum(compute_R_times), R))
        t_compute_r_end = time.time()
        tf.logging.info("Master done computing R... Elapsed time: %f" % (t_compute_r_end-t_compute_r_start))
        compute_R_times.append(t_compute_r_end-t_compute_r_start)
      if FLAGS.should_compute_R and FLAGS.task_id != 0:
        distributed_compute_R(mon_sess)

      cur_epoch_track = max(cur_epoch_track, new_epoch_track)

      # Dequeue variable batchsize inputs
      images_real, labels_real = mon_sess.run(variable_batchsize_inputs[FLAGS.batch_size], feed_dict={images:np.zeros([1, 32, 32, 3]), labels: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})
      loss_value, step = mon_sess.run([train_op, global_step], run_metadata=run_metadata, options=run_options, feed_dict={images:images_real,labels:labels_real})
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

      cur_epoch = n_examples_processed / float(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
      tf.logging.info("epoch: %f time %f" % (cur_epoch, time.time()-begin_time));
      if cur_epoch >= FLAGS.n_train_epochs:
        break

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
