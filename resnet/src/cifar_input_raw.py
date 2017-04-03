# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CIFAR dataset input module.
"""

import tensorflow as tf

import os
import re
import sys
import tarfile

from six.moves import urllib


FLAGS = tf.app.flags.FLAGS

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=50000

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' if dataset_name == 'cifar10' else 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-py')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]

def load_cifar_data_raw():
    print("Loading raw cifar10 data...")
    datadir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-py')
    train_filenames = [os.path.join(datadir, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filenames = [os.path.join(datadir, 'test_batch')]

    batchsize = 10000
    train_images, train_labels = [], []
    for x in train_filenames:
        data = unpickle(x)
        images = data["data"].reshape((batchsize, 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = np.array(data["labels"]).reshape((batchsize,))
        train_images += [(crop_center(x, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE)-128.0)/255.0 for x in images]
        train_labels += [x for x in labels]

    test_images, test_labels = [], []
    for x in test_filenames:
        data = unpickle(x)
        images = data["data"].reshape((batchsize, 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = np.array(data["labels"]).reshape((batchsize,))
        test_images += [(crop_center(x, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE)-128.0)/255.0 for x in images]
        test_labels += [x for x in labels]

    print("Done")

    return tuple([np.array(x) for x in [train_images, train_labels, test_images, test_labels]])

def placeholder_inputs():
  IMAGE_SIZE = 32
  images_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
  labels_placeholder = tf.placeholder(tf.int64, shape=(None, 10 if FLAGS.dataset == 'cifar10' else 100))
  return images_placeholder, labels_placeholder

def next_batch_indices(target_batch_size, n_elements, cur_index, exclude_index=-1, swap_index=-1):
    indices = list(range(cur_index, min(n_elements, cur_index + target_batch_size)))
    next_index = cur_index + target_batch_size
    if exclude_index in indices:
        indices.remove(exclude_index)
    indices = [exclude_index if x == swap_index else x for x in indices]
    while next_index < n_elements and len(indices) < target_batch_size:
        indices.append(cur_index)
        next_index += 1
    if next_index >= n_elements:
        next_index = 0
    return indices, next_index

def next_batch(target_batch_size, images, labels, cur_index, exclude_index=-1, swap_index=-1):
    indices, next_index = next_batch_indices(target_batch_size, len(images), cur_index, exclude_index, swap_index)
    assert(len(indices) != 0)
    return images[indices], labels[indices], next_index
