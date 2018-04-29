from functools import partial

import numpy as np
import tensorflow as tf


def masked_softmax(logits, mask, name=None):
    negative_mask = mask - 1
    logits = logits + 99999 * negative_mask
    return tf.nn.softmax(logits, name=name)


def masked_softmax_cross_entropy_with_logits(logits, mask, labels):
    negative_mask = mask - 1
    logits = logits + 99999 * negative_mask
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)


def get_nonl(nonl_name):
    if nonl_name == 'relu':
        nonl = tf.nn.relu
    elif nonl_name == 'tanh':
        nonl = tf.nn.tanh
    else:
        raise NotImplementedError("Nonlinearity {0} not implemented".format(nonl_name))
    return nonl


def get_optimizer(optimizer_name, global_step, cost, lr_boundaries, lr_values, params):
    learning_rate = tf.train.piecewise_constant(
        global_step, lr_boundaries, lr_values)
    if optimizer_name == 'momentum':
        train_op = tf.train.MomentumOptimizer(learning_rate, params['momentum']).minimize(
            cost, global_step=global_step)
    else:
        raise NotImplementedError("Optimizer {0} not implemented".format(optimizer_name))
    return train_op


def get_policy_head(name, input, normalization, nonl, params):
    if name == 'az_stone':
        return az_stone_policy_head(input, normalization, nonl, **params)
    raise NotImplementedError


def get_value_head(name, input, normalization, nonl, params):
    if name == 'az':
        return az_value_head(input, normalization, nonl, **params)
    raise NotImplementedError


def az_stone_policy_head(input, normalization, nonl, boardsize, actionsize):
    """
    A Policy Head for stone placement games.
    Matches AlphaZero in Go
    Uses a flat representation of the actions, as used in Go
    :param input: the output of the last convolution layer
    :param normalization: The layerwise normalization being used
    :param nonl: The nonlinearity being used
    :param boardsize: The size n of the n x n board.
    :param actionsize: The number of legal actions possible in the game.
    :return: logits: The logits of the policy (unmasked)
    """
    net = nonl((normalization(tf.layers.conv2d(input, filters=2, kernel_size=[1, 1], padding="same"))))
    logits = tf.layers.dense(tf.reshape(net, [-1, boardsize * boardsize * 2]), actionsize)
    return logits


def az_value_head(input, normalization, nonl, boardsize, fc_width):
    net = nonl((normalization(tf.layers.conv2d(input, filters=1, kernel_size=[1, 1], padding="same"))))
    net = nonl(tf.layers.dense(tf.reshape(net, [-1, boardsize * boardsize]), fc_width))
    logit = tf.layers.dense(net, 1)
    return logit


def resnet_model_fn(features, labels, mode, params):
    """
    :param features: must contain 'input', may contain 'mask'
    :param labels: must contain 'policy' and 'value'
    :param mode: a tf.estimator.ModeKeys
    :param params: dict of parameters
        batchn : bool (whether to use batch normalisation)
        channels : int (number of channels in the residual network stack
        stack_depth : int (depth of stack)
        nonlinearity : string (name of the nonlinearity to use)
        mask : bool (whether there is a mask for illegal actions. If so, features[0] is the network input, features[1]
                     is the mask)
        policy_head : policy head name from hydra
        policy_head_params : a dict of params
        value_head : value head name from hydra
        value_head_params : a dict of params
        value_nonl : nonlinearity used for value function
        l2_strength : constant strength of l2 regularization
        optimizer_name : e.g. 'momentum', 'sgd', 'adam'
        optimizer_params : dictionary of params
    """
    # Set up layer functions
    if params['batchn']:
        my_normalisation = partial(tf.layers.batch_normalization,
                                   momentum=.997, epsilon=1e-5, fused=True, center=True, scale=True,
                                   training=(mode == tf.estimator.ModeKeys.TRAIN))
    else:
        my_normalisation = lambda net: net

    my_conv2d = partial(tf.layers.conv2d,
                        filters=params['channels'], kernel_size=[3, 3], padding="same")

    my_nonl = get_nonl(params['nonlinearity'])

    def my_res_layer(inputs):
        net = my_normalisation(my_conv2d(inputs))
        net = my_nonl(net)
        net = my_normalisation(my_conv2d(net))
        net = my_nonl(net + inputs)
        return net

    # We now build the body of the network. We accept input via features, or features['input']
    try:
        net = my_nonl(my_normalisation(my_conv2d(features['input'])))
    except TypeError:
        net = my_nonl(my_normalisation(my_conv2d(features)))

    # resnet stack
    for _ in range(params['stack_depth']):
        net = my_res_layer(net)

    # We now have the shared representation to go into our heads.
    # We hard code having a policy head and a value head
    shared_rep = net

    # # Policy head
    # policy_logits = get_policy_head(params['policy_head'], shared_rep, my_normalisation, my_nonl,
    #                                 params['policy_head_params'])

    # if params['mask']:
    #     policy = masked_softmax(policy_logits, features['mask'], name='policy_output')
    # else:
    #     policy = tf.nn.softmax(policy_logits, name='policy_output')

    # Value Head
    value_activations = get_value_head(params['value_head'], shared_rep, my_normalisation, my_nonl,
                                       params['value_head_params'])
    value_output = get_nonl(params['value_nonl'])(value_activations, name='value_output')

    if mode == tf.estimator.ModeKeys.PREDICT:
        # return {'policy': policy, 'value': value_output}
        return {'value': value_output}

    # Training & Metrics
    global_step = tf.train.get_or_create_global_step()
    # if params['mask']:
    #     policy_cost = masked_softmax_cross_entropy_with_logits(policy_logits, features['mask'],
    #                                                                  labels['policy'])
    # else:
    #     policy_cost = tf.nn.softmax_cross_entropy_with_logits(logits=policy_logits, labels=labels['policy'])
    # policy_cost = tf.reduce_mean(policy_cost)

    value_cost = tf.reduce_mean(tf.square(value_output - labels['value']))

    l2_cost = params['l2_strength'] * tf.add_n([tf.nn.l2_loss(v)
                                                for v in tf.trainable_variables() if not 'bias' in v.name])

    # combined_cost = policy_cost + value_cost + l2_cost
    combined_cost = value_cost + l2_cost

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = get_optimizer(params['optimizer_name'], global_step, combined_cost,
                                 params['lr_boundaries'], params['lr_values'],
                                 params['optimizer_params'])

    # policy_entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.log(policy), axis=1))
    # metric_ops = {
    #     'accuracy': tf.metrics.accuracy(labels=labels['policy'],
    #                                     predictions=policy,
    #                                     name='accuracy_op'),
    #     'policy_cost': tf.metrics.mean(policy_cost),
    #     'value_cost': tf.metrics.mean(value_cost),
    #     'l2_cost': tf.metrics.mean(l2_cost),
    #     'policy_entropy': tf.metrics.mean(policy_entropy),
    #     'combined_cost': tf.metrics.mean(combined_cost),
    # }
    # Create summary ops so that they show up in SUMMARIES collection
    # That way, they get logged automatically during training

    # for metric_name, metric_op in metric_ops.items():
    #     tf.summary.scalar(metric_name, metric_op[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        raise NotImplementedError()

    # return train_op, (combined_cost, policy_cost, value_cost, l2_cost)
    return train_op, (combined_cost, value_cost, l2_cost)


class NotEstimator(object):
    """
    Like an Estimator, but not
    """
    def __init__(self, model_fn, params):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.model_fn = model_fn
        self.params = params
        self.input_shape = [None, params['boardsize'], params['boardsize'], 3]
        self.mask_shape = [None, params['policy_head_params']['actionsize']]

        self.inference_input = None
        self.inference_mask = None
        self.inference_output = None
        self.filename_placeholder = None
        self.train_iterator = None
        self.train_op = None
        self.losses = None

    def initialize_graph(self):
    # def initialize_graph(self, mode, train_input_function=None):
        with self.graph.as_default():
            # if mode == tf.estimator.ModeKeys.PREDICT:
            self.inference_input = tf.placeholder(tf.float32, self.input_shape)
            self.inference_mask = tf.placeholder(tf.float32, self.mask_shape)

            self.inference_output = self.model_fn({'input': self.inference_input, 'mask': self.inference_mask},
                                                  None, tf.estimator.ModeKeys.PREDICT, self.params)
            self.sess.run(tf.global_variables_initializer())
            # else:
            #     self.filename_placeholder = tf.placeholder(tf.string, shape=[None])
            #     self.train_iterator = train_input_function(self.filename_placeholder)
            #     features, labels = self.train_iterator.get_next()
            #     self.train_op, self.losses = self.model_fn(features, labels, tf.estimator.ModeKeys.TRAIN, self.params)
            #     self.sess.run(tf.global_variables_initializer())

    def set_weights(self, ray_tf_variables):
        with self.graph.as_default():
            all_out = tf.group([self.inference_output[key] for key in self.inference_output])
            variables = ray.experimental.tfutils.TensorFlowVariables(all_out, self.sess)
            variables.set_weights(ray_tf_variables)

    def predict(self, numpy_array_inputs, numpy_array_mask):
        with self.graph.as_default():
            outputs = self.sess.run(self.inference_output, feed_dict={self.inference_input: numpy_array_inputs,
                                                                      self.inference_mask: numpy_array_mask})
        return outputs

    def train(self, filenames):
        with self.graph.as_default():
            self.sess.run(self.train_iterator.initializer, feed_dict={self.filename_placeholder: filenames})
            step = 0
            while True:
                try:
                    _, loss_value = self.sess.run((self.train_op, self.losses))
                    print('loss is {0} on step {1}'.format(loss_value, step))
                    step += 1
                except tf.errors.OutOfRangeError:
                    break

            variables = ray.experimental.tfutils.TensorFlowVariables(self.losses[0], self.sess)
            return variables.get_weights()
