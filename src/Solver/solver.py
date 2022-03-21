#_*_coding:utf-8_*_

import tensorflow as tf
import numpy as np
tf.set_random_seed(12345)
np.random.seed(12345)
import os
import logging

from Utils import utils

from sklearn.metrics import roc_auc_score


def train_and_evaluate(input_fn_train, input_fn_eval, model_fn, params):
  """
  Train the model and evaluate every epoch
  :param input_fn: (BaseData) input function
  :param model_fn: (BaseModel) model function
  :param params: (BaseParams) hyper-parameters
  """

  # creating model
  input_fn_train.shuffle_dataset()
  train_inputs = input_fn_train.get_data('train')
  eval_inputs = input_fn_eval.get_data('test')
  model_spec_train = model_fn(train_inputs, is_training=True)
  model_spec_eval = model_fn(eval_inputs, is_training=False)

  # creating tf.Saver instances to save weights during training
  last_saver = tf.train.Saver(max_to_keep=params.keep_ckpt_max)
  best_saver = tf.train.Saver(max_to_keep=1)  # keep best checkpoint on eval.

  config = tf.ConfigProto(inter_op_parallelism_threads=8,
                          intra_op_parallelism_threads=8,
                          allow_soft_placement=True)
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # For tensorboard
    train_writer = tf.summary.FileWriter(
      os.path.join(params.model_dir, 'train_summaries'), sess.graph
    )
    eval_writer = tf.summary.FileWriter(
      os.path.join(params.model_dir, 'eval_summaries'), sess.graph
    )

    best_eval_metric = 0.0
    max_stop_num = 3
    early_stop_num = 0
    
    sess.run(model_spec_eval['iterator_init_op'])
    for epoch in range(params.num_epochs):
      # train one epoch on training set
      logging.info('Epoch {}/{}'.format(epoch, params.num_epochs))
      train_metrics = train_sess(sess, model_spec_train, params, epoch)
      print(train_metrics)
      # save model weights
      last_save_path = os.path.join(params.model_dir, 'last_model/model')
      last_saver.save(sess, last_save_path, global_step=epoch+1)

      # evaluate one epoch on validation set
      valid_metrics = evaluate_sess(sess, model_spec_eval, params)
      print(valid_metrics)
      input_fn_train.shuffle_dataset()


def train_sess(sess, model_spec, params, epoch):
  # Get relevant graph operations or nodes needed for training
  loss_op = model_spec['loss']
  train_op = model_spec['train_op']
  preds_op = model_spec['preds']
  labels_op = model_spec['label']
  group_key_op = model_spec['global_group_id']
  global_step_op = tf.train.get_or_create_global_step()

  # re-initialize the training dataset iterator
  sess.run(model_spec['iterator_init_op'])
  step = 0
  preds_dict = {}
  step_avg_loss = 0.0
  while True:
    step += 1
    try:
      _, loss, preds, labels, group_key = sess.run([
          train_op, loss_op, preds_op, labels_op, group_key_op 
      ])

      step_avg_loss += loss
      if step % params.show_log_steps == 0:
        step_avg_loss /= params.show_log_steps
        logging.info('step-{}: loss {:.4f}'.format(step, step_avg_loss))
        step_avg_loss = 0.0

      for i in range(len(group_key)):
        if group_key[i] not in preds_dict:
          preds_dict[group_key[i]] = []
        preds_dict[group_key[i]].append((labels[i], preds[i], loss))
    except tf.errors.OutOfRangeError:
      logging.info('train session is over....')
      break
  np.save('pred_train.npy', preds_dict)
  metrics = evaluate_results(preds_dict)
  return metrics


def evaluate_sess(sess, model_spec, params):
  preds_op = model_spec['preds']
  labels_op = model_spec['label']
  group_key_op = model_spec['global_group_id']
  loss_op = model_spec['log_loss']
  global_step_op = tf.train.get_or_create_global_step()

  # re-initialize the evaluation dataset iterator
  sess.run(model_spec['iterator_init_op'])

  # evaluation
  step = 0
  preds_dict = {}
  while True:
    step += 1
    try:
      preds, group_key, labels, loss = sess.run([
          preds_op, group_key_op, labels_op, loss_op
      ])
      for i in range(len(group_key)):
        if group_key[i] not in preds_dict:
          preds_dict[group_key[i]] = []
        preds_dict[group_key[i]].append((labels[i], preds[i], loss))
    except tf.errors.OutOfRangeError:
      logging.info('evaluate session is over....')
      break
  np.save('pred_valid.npy', preds_dict)
  metrics = evaluate_results(preds_dict)
  return metrics


def evaluate_results(preds_dict):
  ndcg_dict = {}
  top_k_list = [1, 5, 10]
  for top_k in top_k_list:
    ndcg_dict[top_k] = []
  gauc_list = []
  roc_auc_list = []

  for group_key in preds_dict:
    user_metrics = utils.compute_metrics(preds_dict[group_key], top_k_list)
    if user_metrics is None:
      continue
    roc_auc_list.extend(preds_dict[group_key])
    num_samples = len(preds_dict[group_key])
    gauc_list.append((user_metrics['gauc'], num_samples))
    for top_k in top_k_list:
      ndcg_dict[top_k].append((
        user_metrics['ndcg@{}'.format(top_k)],
        num_samples
      ))

  def metric_template(arrs):
    arrs = np.asarray(arrs)
    metric_k = sum(arrs[:, 0] * arrs[:, 1]) / sum(arrs[:, 1])
    return metric_k

  metrics = {}
  for top_k in top_k_list:
    ndcg_k = metric_template(ndcg_dict[top_k])
    metrics['ndcg@{}'.format(top_k)] = round(ndcg_k, 4)
  gauc = metric_template(gauc_list)
  metrics['gauc'] = round(gauc, 4)
  labels, preds, loss = zip(*roc_auc_list)
  metrics['auc'] = round(roc_auc_score(labels, preds), 4)
  metrics['loss'] = np.mean(loss)
  return metrics



