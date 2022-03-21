#_*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
tf.set_random_seed(12345)
np.random.seed(12345)

from Utils import utils
import logging
import copy


class CimModel():
  def __init__(self, params):
    self.params = params
    global_step = tf.train.get_or_create_global_step()
    self.learning_rate = 0.001

  def __call__(self, inputs, is_training, reuse=False):
    params = copy.deepcopy(self.params)
    mode = 'train' if is_training else 'valid'
    self.is_training =  is_training
    params.dropout_prob = 0.0
    tf.summary.scalar('dropout_prob', params.dropout_prob)
    self.labels = tf.cast(inputs['label'], tf.float32)

    with tf.variable_scope('Rank', reuse=tf.AUTO_REUSE):
      with tf.variable_scope('embedding_layer'):
        self.build_embedding_layer(params)

      with tf.variable_scope('input_layer'):
        self.build_input_layer(inputs, params)

      with tf.variable_scope('ranking_layer'):
        self.build_ranking_layer(params)

      with tf.variable_scope('loss_layer'):
        self.build_loss_layer(params)

      model_spec = inputs
      model_spec['preds'] = self.preds
      model_spec['loss'] = self.loss
      model_spec['log_loss'] = self.log_loss

      self.build_train_operation(params)
      model_spec['train_op'] = self.train_op
      
      return model_spec

  def build_embedding_layer(self, params):
    self.sid_embeddings = utils.build_embeddings(
      emb_name='sku_id_embeddings',
      vocab_size=params.num_skus,
      emb_dim=params.emb_dim
    )
    self.cid_embeddings = utils.build_embeddings(
      emb_name='cate_id_embeddings',
      vocab_size=params.num_cates,
      emb_dim=params.emb_dim
    )
    self.vid_embeddings = utils.build_embeddings(
      emb_name='vender_id_embeddings',
      vocab_size=params.num_venders,
      emb_dim=params.emb_dim
    )
    self.bid_embeddings = utils.build_embeddings(
      emb_name='brand_id_embeddings',
      vocab_size=params.num_brands,
      emb_dim=params.emb_dim
    )
    self.qid_embeddings = utils.build_embeddings(
      emb_name='query_embeddings',
      vocab_size=params.num_queries,
      emb_dim=params.emb_dim
    )
    self.uid_embeddings = utils.build_embeddings(
      emb_name='user_embeddings',
      vocab_size=params.num_users,
      emb_dim=params.emb_dim
    )
    self.price_embeddings = utils.build_embeddings(
      emb_name='price_embeddings',
      vocab_size=params.num_prices,
      emb_dim=params.emb_dim
    )

  def build_input_layer(self, inputs, params):
    self.sid = inputs['sid']
    sid_emb = tf.nn.embedding_lookup(self.sid_embeddings, inputs['sid'])
    vid_emb = tf.nn.embedding_lookup(self.vid_embeddings, inputs['vid'])
    cid_emb = tf.nn.embedding_lookup(self.cid_embeddings, inputs['cid'])
    bid_emb = tf.nn.embedding_lookup(self.bid_embeddings, inputs['bid'])
    price_emb = tf.nn.embedding_lookup(self.price_embeddings, inputs['price'])

    item_emb = tf.concat([sid_emb, vid_emb, cid_emb, bid_emb, price_emb], axis=1)
    self.item_emb = tf.reshape(item_emb, [-1, params.emb_dim * 5])

    qid_emb = tf.nn.embedding_lookup(self.qid_embeddings, inputs['qid'])
    self.query_emb = tf.reshape(qid_emb, [-1, params.emb_dim])

    uid_emb = tf.nn.embedding_lookup(self.uid_embeddings, inputs['uid'])
    self.user_id_emb = tf.reshape(uid_emb, [-1, params.emb_dim])
    self.user_emb = self.user_id_emb

    self.group_id = inputs['group_id']  # [real_batch_size]
    self.global_group_id = inputs['global_group_id']
    
    req_ids = tf.reshape(inputs['req'], [-1, params.max_rq_len, 5])
    req_sid = req_ids[:, :, 0]
    req_cid = req_ids[:, :, 1]
    req_bid = req_ids[:, :, 2]
    req_vid = req_ids[:, :, 3]
    req_price = req_ids[:, :, 4]
    sid_rq_emb = tf.nn.embedding_lookup(self.sid_embeddings, req_sid)
    cid_rq_emb = tf.nn.embedding_lookup(self.cid_embeddings, req_cid)
    bid_rq_emb = tf.nn.embedding_lookup(self.bid_embeddings, req_bid)
    vid_rq_emb = tf.nn.embedding_lookup(self.vid_embeddings, req_vid)
    price_rq_emb = tf.nn.embedding_lookup(self.price_embeddings, req_price)
    self.rq_emb = tf.reshape(
      tf.concat([sid_rq_emb, vid_rq_emb, cid_rq_emb, bid_rq_emb, price_rq_emb], axis=-1),
      [-1, params.max_rq_len, params.emb_dim * 5]
    )
    self.rq_mask = tf.to_float(inputs['req_mask'])
    self.rq_mask = tf.reshape(self.rq_mask, [-1, params.max_rq_len])
    self.group_id = inputs['group_id']
    self.sid_rq = req_sid
    self.max_rq_len = params.max_rq_len

    group_len = inputs['group_len']
    group_mask = tf.sequence_mask(group_len, params.max_group_len, tf.float32)
    group_mask = tf.reshape(group_mask, [-1, params.max_group_len])
    self.group_mask = group_mask

  def get_request_score(self, user_query_emb, request_emb):
    user_query_emb = tf.reshape(user_query_emb, [-1, 1, 16 * 7])
    user_query_emb = tf.tile(user_query_emb, [1, self.max_rq_len, 1])
    user_query_request = tf.concat([user_query_emb, request_emb], axis=-1)
      
    layer_1 = tf.layers.dense(user_query_request, 256, activation=tf.nn.relu,
                                  name='request_score_1', reuse=tf.AUTO_REUSE)
    layer_2 = tf.layers.dense(layer_1, 128, activation=tf.nn.relu,
                                  name='request_score_2', reuse=tf.AUTO_REUSE)
    request_score = tf.layers.dense(layer_2, 1, name='request_score_3', reuse=tf.AUTO_REUSE)
    request_score = tf.reshape(request_score, [-1, self.max_rq_len])
    return request_score
      
  def get_request_point_loss(self, request_label, request_score):
    request_label = tf.reshape(request_label, [-1, 1])
    request_score = tf.reshape(request_score, [-1, 1])
    loss_point = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=request_score, labels=request_label
    ))
    return loss_point

  def get_request_label(self, sku_ids, request_ids):
    sku_ids = tf.reshape(sku_ids, [-1, 1])
    request_ids = tf.nn.embedding_lookup(request_ids, self.group_id)
    cond = tf.equal(sku_ids, request_ids)
    cond = tf.to_float(cond)
    request_label = tf.segment_sum(cond, self.group_id)
    request_label = tf.clip_by_value(request_label, 0.0, 1.0)
    request_label = tf.nn.embedding_lookup(request_label, self.group_id)
    return request_label


  def build_ranking_layer(self, params):
    # override this function for customized model
    request_emb = tf.nn.embedding_lookup(tf.reshape(self.rq_emb, [-1, self.max_rq_len * 16 * 5]), self.group_id)
    request_emb = tf.reshape(request_emb, [-1, self.max_rq_len, 16 * 5])
    user_item_emb = tf.concat([self.user_emb, self.item_emb, self.query_emb], -1) 
    request_score = self.get_request_score(user_item_emb, request_emb)
    paddings = tf.ones_like(request_score) * (-2 ** 31 + 1)
    rq_mask = tf.nn.embedding_lookup(self.rq_mask, self.group_id)
    request_score =  tf.where(tf.greater(rq_mask, 0), request_score, paddings)
    request_weights = tf.nn.sigmoid(request_score * 1, name="request_weights")

    interaction_emb = utils.cie_module(
          query_emb=user_item_emb,
          context_emb=request_emb,
          mask=rq_mask,
          params=params,
          group_id=self.group_id,
          request_weights=tf.reshape(request_weights, [-1, self.max_rq_len, 1])
    )
    b_s = tf.shape(self.sid_rq)[0]
    request_label = self.get_request_label(self.sid, self.sid_rq)
    top_index_bias = tf.range(0, b_s * self.max_rq_len, self.max_rq_len)
    top_index_bias = tf.reshape(top_index_bias, [b_s, 1])
      
    request_loss_point = self.get_request_point_loss(request_label, request_score)
    self.aux_loss = request_loss_point

    mlp_input = tf.concat([self.user_emb, self.item_emb, self.query_emb, interaction_emb], axis=1)
    mlp_output = utils.mlp(input=mlp_input, params=params)
    logits = tf.squeeze(tf.layers.dense(mlp_output, 1, name='pred'))
    preds = tf.nn.sigmoid(logits)
    self.logits = logits
    self.preds = preds

  def build_train_operation(self, params, train_vars=None):
    for var in tf.trainable_variables():
      logging.info('{}, {}'.format(var.name, var.shape))
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-05, use_locking=True)

    tvars = tf.trainable_variables()
    grads = tf.gradients(self.loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)
    self.train_op = train_op


  def build_loss_layer(self, params):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=self.logits, labels=self.labels
    ))

    self.loss = loss + self.aux_loss 
    self.log_loss = loss


