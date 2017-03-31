
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

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
import inspect
import time

import numpy as np
import tensorflow as tf

import reader

np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
logging = tf.logging

tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
flags.DEFINE_string(
    "model", "distributed",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("should_evaluate", False,
                  "Evaluate on training data after epochs")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')
flags.DEFINE_integer(
    'batch_size', 0, 'Batch size')
flags.DEFINE_string('ps_hosts', '',
                    """Comma-separated list of hostname:port for the """
                    """parameter server jobs. e.g. """
                    """'machine1:2222,machine2:1111,machine2:2222'""")
flags.DEFINE_string('worker_hosts', '',
                    """Comma-separated list of hostname:port for the """
                    """worker jobs. e.g. """
                    """'machine1:2222,machine2:1111,machine2:2222'""")
flags.DEFINE_string('job_name', '',
                    "worker or ps")
flags.DEFINE_string('train_dir', '/tmp/lstm_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")


FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_, gstep):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    self.global_step = gstep

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    embedding = tf.get_variable(
        "embedding", [vocab_size, size], dtype=data_type())
    inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)

    num_workers, num_replicas_to_aggregate = len(FLAGS.worker_hosts.split(",")), FLAGS.num_replicas_to_aggregate
    tf.logging.info("Num to aggregate: %d" % num_replicas_to_aggregate)
    if num_replicas_to_aggregate == -1:
        num_replicas_to_aggregate = num_workers

    optimizer = tf.train.GradientDescentOptimizer(self._lr)

    optimizer = tf.train.SyncReplicasOptimizer(
      optimizer,
      replicas_to_aggregate=num_replicas_to_aggregate,
      total_num_replicas=num_workers,
    )
    self.opt = optimizer

    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
      global_step=self.global_step)

    #self._train_op = optimizer.apply_gradients(
    #  zip(grads, tvars))


    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

class DistributedConfig(object):
  """Like medium config, except infinite epochs, no decay"""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 1000000000
  keep_prob = 1
  lr_decay = .5
  batch_size = FLAGS.batch_size
  vocab_size = 10000

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    tf.logging.info("Evaluating...")
    vals = session.run(fetches, feed_dict)
    tf.logging.info("Done Evaluating...")
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    tf.logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
                    (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                     iters * model.input.batch_size / (time.time() - start_time)))
    sys.stdout.flush()

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  elif FLAGS.model == "distributed":
    return DistributedConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):

  num_workers = len(FLAGS.worker_hosts.split(","))

  assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

  ps_hosts = FLAGS.ps_hosts.split(',')
  worker_hosts = FLAGS.worker_hosts.split(',')
  cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                       'worker': worker_hosts})
  server = tf.train.Server(
      {'ps': ps_hosts,
       'worker': worker_hosts},
      job_name=FLAGS.job_name,
      task_index=FLAGS.task_id)

  if FLAGS.job_name == 'ps':
    # `ps` jobs wait for incoming connections from the workers.
    server.join()
    return

  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_train_config = get_config()
  eval_train_config.batch_size = 1
  eval_train_config.num_steps = 1

  # Distributed model creation
  with tf.device(
          tf.train.replica_device_setter(
              worker_device='/job:worker/task:%d' % FLAGS.task_id,
              cluster=cluster_spec)):

    global_step = tf.Variable(0, name="global_step", trainable=False)
    global_step_eval = tf.Variable(0, name="global_step_eval", trainable=False)
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input, gstep=global_step)

    with tf.name_scope("EvalTrain"):
      eval_train_input = PTBInput(config=eval_train_config, data=train_data, name="EvalTrain")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        m_eval_train = PTBModel(is_training=False, config=config, input_=eval_train_input, gstep=global_step_eval)

    with ops.device(m.opt._global_step.device):
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


  is_chief = FLAGS.task_id == 0
  sync_replicas_hook_train = m.opt.make_session_run_hook(is_chief)
  checkpoint_save_secs = 60*2
  evaluate_times, compute_R_times = [0], [0]
  n_examples_processed = 0
  cur_epoch_track = 0

  with tf.train.MonitoredTrainingSession(
          master=server.target, is_chief=is_chief,
          hooks=[sync_replicas_hook_train],
          checkpoint_dir=FLAGS.train_dir,
          save_checkpoint_secs=checkpoint_save_secs) as session:

    tf.logging.info("Starting to train...")
    sys.stdout.flush()

    while True:

      session.run([workers_block_if_necessary_op])

      new_epoch_float = n_examples_processed / float(m.input.epoch_size)
      new_epoch_track = int(new_epoch_float)

      if FLAGS.should_evaluate and FLAGS.task_id == 0 and (cur_iteration == 0 or new_epoch_track == cur_epoch_track+1):
          session.run([block_workers_op])
          t_evaluate_start = time.time()
          eval_train_perplexity = run_epoch(session, m_eval_train)
          t_evaluate_end = time.time()
          # The second to last number that is printed is 0 (which is usually accuracy in mnist and resnet).
          # This is because there is no accuracy in ptb.
          tf.logging.info("IInfo: %f %d %f %f" % (t_evaluate_start-sum(evaluate_times)-sum(compute_R_times), i + 1, 0, eval_train_perplexity))
          sys.stdout.flush()
          evaluate_times.append(t_evaluate_end-t_evaluate_start)
          session.run([unblock_workers_op])

      cur_epoch_track = max(cur_epoch_track, new_epoch_track)

      # Learning rate decay, which is nil for distributed training...
      m.assign_lr(session, config.learning_rate)

      ####
      # Optimization
      ###
      state = session.run(m.initial_state)
      feed_dict = {}
      for i, (c, h) in enumerate(m.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h
      tf.logging.info("Evaluating...")
      session.run([m.train_op])
      tf.logging.info("Done Evaluating...")
      n_examples_processed += FLAGS.batch_size * num_workers
      ####

    if FLAGS.save_path:
      tf.logging.info("Saving model to %s." % FLAGS.save_path)
      sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
