#_*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import json

from pprint import pprint

class BaseParams(object):
  """
  class that loads hyper-parameters from a json file.
  Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5 # change the value of learning rate in params
  """

  def __init__(self, args):
    """Loads parameters from json file"""

    model_name = args.model_name
    general_params_path = os.path.join(
      args.project_dir, 'src/Config/general_params.json')
    with open(general_params_path) as f:
      params = json.load(f)

    # pre-processing
    params['output_dir'] = os.path.join(args.project_dir, 'models')
    params['data_dir'] =os.path.join(args.project_dir, 'data')

    save_ckpt_steps = params['train_data_len'] // params['batch_size']
    num_train_steps = save_ckpt_steps * params['num_epochs']
    new_params = {
      # 'save_ckpt_steps': save_ckpt_steps,
      'num_train_steps': num_train_steps,  # for learning rate decay
      'model_name': model_name,
      'model_dir_list': []
    }
    params.update(new_params)
    if params['mode'] == 'train':
      params['model_date'] = time.strftime('%Y%m%d-%H%M%S')

    # initialize variables of class, i.e., self.model_name = model_name
    self.__dict__.update(params)
    self.__dict__['model_dir_list'].extend([
      self.__dict__['dropout_prob'],
      self.__dict__['agg_layers'],
      self.__dict__['emb_dim']
    ])

  def update(self, dict_param):
    self.__dict__.update(dict_param)

  def save(self):
    """Saves parameters to json file"""
    json_path = os.path.join(self.__dict__['model_dir'], 'params.json')
    with open(json_path, 'w') as f:
      json.dump(self.__dict__, f, indent=2)

  def make_model_dir(self):
    """update parameters"""
    sub_dir = '{}/{}'.format(
      self.__dict__['model_name'],
      #self.__dict__['model_date'],
      '_'.join(map(str, self.__dict__['model_dir_list'][::-1]))
    )
    model_dir = os.path.join(self.__dict__['output_dir'], sub_dir)

    if self.__dict__['mode'] == 'train':
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    self.__dict__['model_dir'] = model_dir

    self.init_logger()

  @property
  def dict(self):
    """
    Gives dict-like access to Params instance by `params.dict['learning_rate']`
    """
    return self.__dict__

  def init_logger(self):
    """
    Sets the logger to log info in terminal and file `log_path`.
    :return:
    """
    log_path = os.path.join(self.__dict__['model_dir'], 'train_log.txt')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
      # Logging to a file
      file_handler = logging.FileHandler(log_path)
      file_handler.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
      logger.addHandler(file_handler)

      # Logging to console
      stream_handler = logging.StreamHandler()
      stream_handler.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
      logger.addHandler(stream_handler)

    for arg in self.__dict__:
      logging.info('{}: {}'.format(arg, self.__dict__.get(arg)))

