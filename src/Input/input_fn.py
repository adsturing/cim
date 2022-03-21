# _*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle
import copy
import random
random.seed(12345)
import json

import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf

tf.set_random_seed(12345)
np.random.seed(12345)
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.append('./src')
from Utils import utils
from Utils.params_config import *


class BaseData(object):
  def __init__(self, params, filename):
    self.params = params
    output_types = {key: tf.int32 for key in params.feature_name.split(',')}
    output_types.update({'group_id': tf.int32})
    output_types.update({'group_index': tf.int32})
    output_types.update({'req_mask': tf.int32})
    output_types.update({'group_len': tf.int32})
    output_types.update({'global_group_id': tf.int32})
    self.output_types = output_types
    self.group_data = self.load_dataset_from_disk(filename)
    self.data_size = len(self.group_data)
    self.indexes = list(range(self.data_size))
    self.max_rq_len = params.max_rq_len
    print('data size is : {}'.format(self.data_size))

  def get_data(self, mode):
    logging.info('creating {} tf.data.dataset instance...'.format(mode))

    dataset = tf.data.Dataset.from_generator(
      lambda: self.data_generator(),
      output_types=self.output_types
    )

    dataset = dataset.batch(1).prefetch(self.params.buffer_size)

    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    # iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    features = {key: value[0] for key, value in features.items()}
    features['iterator_init_op'] = init_op

    return features

  def shuffle_dataset(self):
    random.shuffle(self.indexes)

  def load_dataset_from_disk(self, filename):
    logging.info('loading {} from disk...'.format(filename))
    path = os.path.join(self.params.data_dir, filename)
    with open(path, 'rb') as reader:
      group_data = pickle.load(reader)
    return group_data

  def data_generator(self):
    batch_size = self.params.batch_size
    feature_name = self.params.feature_name.split(',')
    global_group_id = 0
    for i in range(0, self.data_size, batch_size):
      batch_data = defaultdict(list)
      start = i
      end = self.data_size if i + batch_size > self.data_size else i + batch_size
      group_id, group_start = 0, 1
      for j in range(start, end):
        line = self.group_data[self.indexes[j]]
        user, query = line[0], line[1]
        ori_group_size = len(line[2])
        if ori_group_size > self.params.max_group_len:
            group_size = self.params.max_group_len
        else:
            group_size = ori_group_size
        user_query = [[user for _ in range(group_size)], [query for _ in range(group_size)]]

        for k, key in enumerate(feature_name):
          if k < 2:
            batch_data[key].extend(user_query[k])
          elif k == 2:
            label_info = line[k]
            batch_data[key].extend(label_info[:group_size])
          else:
            break
        expo_info = line[3:3+group_size]
        for k in range(group_size):
          for key, value in zip(feature_name[3:], expo_info[k]):
            batch_data[key].append(value)
        req_data = sum(line[3+ori_group_size], [])
        max_req_data_size = 5 * self.max_rq_len
        if len(line[3+ori_group_size]) > self.max_rq_len:
          req_mask = [1 for _ in range(self.max_rq_len)]
        else:
          req_mask = [1 for _ in range(len(line[3+ori_group_size]))] + \
              [0 for _ in range(self.max_rq_len - len(line[3+ori_group_size]))]
        if len(req_data) < max_req_data_size:
          req_data = req_data + [0 for _ in range(max_req_data_size - len(req_data))]
        elif len(req_data) > max_req_data_size:
          req_data = req_data[:max_req_data_size]
        batch_data['req'].append(req_data)
        batch_data['group_len'].extend([group_size for _ in range(group_size)])
        batch_data['req_mask'].append(req_mask)
        group_ids = [group_id for _ in range(group_size)]
        batch_data['group_id'].extend(group_ids)
        global_group_ids = [global_group_id for _ in range(group_size)]
        batch_data['global_group_id'].extend(global_group_ids)
        global_group_id += 1
        group_index = [m for m in range(group_start, group_start + group_size)]
        for m in range(group_size, self.params.max_group_len):
          group_index.append(0)
        batch_data['group_index'].extend(group_index)
        group_id += 1
        group_start += group_size
      yield batch_data



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--project_dir', type=str,
                      default='.')
  parser.add_argument('--local', type=str, default='yes')
  parser.add_argument('--model_name', type=str, default='base')
  args = parser.parse_args()
  params = BaseParams(args)
  params.batch_size = 2
  input_fn = BaseData(params, params.train_filename)

  train_inputs = input_fn.get_data('train')
  
  with tf.Session() as sess:
    for i in range(2):
      for _ in range(2):
        sess.run(train_inputs['iterator_init_op'])
        try:
          uid, cid, group_id = sess.run([train_inputs['uid'], train_inputs['sid'], train_inputs['req']])
          print(uid, cid, group_id)
          print(_)
        except tf.errors.OutOfRangeError:
          print('eval session is over....')




