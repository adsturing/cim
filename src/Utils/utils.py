#_*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
tf.set_random_seed(12345)
np.random.seed(12345)
random.seed(12345)
import json
import re
import collections
import six

from sklearn.metrics import roc_auc_score


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def build_embeddings(emb_name,
                     vocab_size,
                     emb_dim,
                     zero_padding=True,
                     initializer=create_initializer(0.02)):
  embeddings = tf.get_variable(
    name=emb_name,
    shape=[vocab_size, emb_dim],
    initializer=initializer
  )
  if zero_padding:
    zero_emb = tf.zeros((1, emb_dim), tf.float32)
    embeddings = tf.concat([zero_emb, embeddings], axis=0)
  return embeddings


def dcg_score(y_true, y_score, k=10, gains="exponential"):
  order = np.argsort(y_score)[::-1]
  y_true = np.take(y_true, order[:k])

  if gains == "exponential":
    gains = 2 ** y_true - 1
  elif gains == "linear":
    gains = y_true
  else:
    raise ValueError("Invalid gains option.")

  discounts = np.log2(np.arange(len(y_true)) + 2)
  return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
  best = dcg_score(y_true, y_true, k, gains)
  actual = dcg_score(y_true, y_score, k, gains)
  return actual / (best + 1e-8)

def compute_metrics(labels_preds, top_k_list):
  labels, preds, loss = zip(*labels_preds)
  labels = list(labels)
  metrics = {}
  num_pos = sum(labels)
  if num_pos == len(labels) or num_pos == 0:
    return None
  else:
    metrics['gauc'] = roc_auc_score(labels, preds)
  for top_k in top_k_list:
    ndcg_k = ndcg_score(labels, preds, k=top_k)
    metrics['ndcg@{}'.format(top_k)] = ndcg_k

  return metrics

def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == 'lrelu':
    return tf.nn.leaky_relu
  elif act == "tanh":
    return tf.tanh
  elif act == "sigmoid":
    return tf.nn.sigmoid
  elif act =='elu':
    return tf.nn.elu
  else:
    raise ValueError("Unsupported activation: %s" % act)


def mlp(input, params):
  """
  Implementation of multi-layer perceptron
  :param input: (Tensor) input of mlp
  :param mlp_layers: (str) layers used in mlp, e.g., '256,128'
  :param dropout_prob: (float) ratio of network to drop
  :param activation: (str) activation function to use
  :return: output of mlp
  """
  act_func = get_activation(params.agg_act)
  mlp_layers = [int(unit) for unit in params.agg_layers.split(',')]
  mlp_net = input
  if params.use_bn:
    mlp_net = tf.layers.batch_normalization(
      mlp_net, training=params.is_training)
    if act_func is not None:
      mlp_net = act_func(mlp_net, name='act_0')

  for i in range(len(mlp_layers)):
    mlp_net = tf.layers.dense(mlp_net, mlp_layers[i], name='fc_{}'.format(i))
    mlp_net = tf.nn.dropout(mlp_net, 1.0 - params.dropout_prob)
    if params.use_bn:
      mlp_net = tf.layers.batch_normalization(
        mlp_net, training=params.is_training)
    if act_func is not None:
      mlp_net = act_func(mlp_net, name='act_{}'.format(i+1))
  return mlp_net


def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    Modify from: https://github.com/pl8787/SetRank/blob/master/src/modules.py   
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable('beta', params_shape,
                               initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', params_shape,
                                initializer=tf.ones_initializer())
        # beta= tf.Variable(tf.zeros(params_shape))
        # gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def multihead_attention_mask(queries, 
                        keys,
                        query_masks,
                        key_masks, 
                        num_units=None, 
                        num_heads=8, 
                        scope="multihead_attention", 
                        reuse=None,
                        activation=tf.nn.relu,
                        request_weights=None):
    '''Applies multihead attention.
    Modify from: https://github.com/pl8787/SetRank/blob/master/src/modules.py   
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      is_training: Boolean. Controller of mechanism for dropout.
      num_heads: An int. Number of heads.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=activation) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=activation) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=activation) # (N, T_k, C)
        if request_weights is not None:
            V = request_weights * V
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs



def cie_module(query_emb, context_emb, mask, params, order=False, group_id=None, request_weights=None):
  _, to_seq_len, emb_dim = context_emb.get_shape().as_list()
  cur_keys = context_emb
  for i in range(params.num_blocks-1):
    with tf.variable_scope("num_blocks_{}".format(i)):
        cur_keys = multihead_attention_mask(
          queries=cur_keys,
          keys=cur_keys,
          query_masks=mask,
          key_masks=mask,
          num_units=emb_dim,
          num_heads=params.num_attention_heads,
          activation=get_activation(params.activation),
          request_weights=request_weights
        )
  query_emb = tf.layers.dense(query_emb, emb_dim, name='query_transform')
  query_emb = tf.reshape(query_emb, [-1, 1, emb_dim])
  _, from_seq_len, _ = query_emb.get_shape().as_list()
  with tf.variable_scope("num_blocks_{}".format(params.num_blocks-1)):
      interaction_emb = multihead_attention_mask(
        queries=query_emb,
        keys=cur_keys,
        query_masks=tf.ones([tf.shape(query_emb)[0], 1]),
        key_masks=mask,
        num_units=emb_dim,
        num_heads=params.num_attention_heads,
        activation=get_activation(params.activation),
        request_weights=request_weights
      )
  return tf.squeeze(interaction_emb, axis=1)

