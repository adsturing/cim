#_*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from Utils.params_config import *
from Input.input_fn import *
from Model.model_fn import *
from Solver.solver import train_and_evaluate 
from Utils import utils

import argparse
import logging
import os

tf.set_random_seed(12345)
np.random.seed(12345)

def process_args():
  # load hyper-parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('--project_dir', type=str,
                      default='.')
  parser.add_argument('--model_name', type=str, default='ours')
  args = parser.parse_args()
  return args


def get_num_info():
  num_info = {
    'num_skus': 15587268,
    'num_venders': 103832,
    'num_cates': 3335,
    'num_brands': 88219,
    'num_queries': 323478,
    'num_users': 931883,
    'num_prices': 452
  }
  return num_info


def main():
  args = process_args()
  params = BaseParams(args)

  num_info = get_num_info()
  params.update(num_info)
  params.make_model_dir()

  # for reproducible results

  # initialize dataset and model
  logging.info('creating datasets....')
  input_fn_train = BaseData(params, params.train_filename)
  input_fn_eval = BaseData(params, params.test_filename)
  model_fn = CimModel(params)

  train_and_evaluate(input_fn_train, input_fn_eval, model_fn, params)

if __name__ == '__main__':
    main()
