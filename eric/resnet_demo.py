#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: resnet-multigpu.py

# to run:
# pip install -U git+https://github.com/ppwwyyxx/tensorpack.git

import sys
import argparse
import numpy as np
import os
from contextlib import contextmanager
import tensorflow as tf

from tensorpack import *
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.utils.stats import RatioCounter
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.collection import freeze_collection
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.varreplace import custom_getter_scope

# from tfbench.convnet_builder import ConvNetBuilder
# from tfbench import model_config

INPUT_SHAPE = 224
IMAGE_DTYPE = tf.uint8
IMAGE_DTYPE_NUMPY = 'uint8'


class RayModel(object):
    def __init__(self):
        self._model = TensorpackModel(data_format="NHWC")
        self.inputs, self.labels = self._model.inputs()

        input_desc = [
            InputDesc.from_placeholder(self.inputs),
            InputDesc.from_placeholder(self.labels)
        ]
        inputs = PlaceholderInput()
        inputs.setup(input_desc)
        with TowerContext('', is_training=True):
            self._model.build_graph(*inputs.get_input_tensors())
        self.loss = self._model.loss

    def feed_dict(self):
        return {
            self.inputs: np.zeros(self.inputs.shape.as_list()),
            self.labels: np.zeros(self.labels.shape.as_list())
        }


class Model(ModelDesc):
    def __init__(self, data_format='NCHW', batch=64):
        self.data_format = data_format
        self.batch = batch

    def inputs(self):
        return [tf.placeholder(IMAGE_DTYPE, [self.batch, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                tf.placeholder(tf.int32, [self.batch], 'label')]

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        return tf.train.GradientDescentOptimizer(lr)

    def build_graph(self, image, label):
        ctx = get_current_tower_context()

        # all-zero tensor hurt performance for some reason.
        label = tf.random_uniform(
            [self.batch],
            minval=0, maxval=1000 - 1,
            dtype=tf.int32, name='synthetic_labels')

        # our fake images are in [0, 1]
        image = tf.cast(image, tf.float32) * 2.0 - 1.
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self._get_logits(image)
        if logits.dtype != tf.float32:
            logger.info("Casting back to fp32 ...")
            logits = tf.cast(logits, tf.float32)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')
        # TODO tensorflow/benchmark only computes WD on 1 GPU.
        if False:
            self.cost = loss    # disable wd
        else:
            wd_cost = regularize_cost('.*', tf.nn.l2_loss) * 1e-4
            self.cost = tf.add_n([loss, wd_cost], name='cost')

        self.loss = self.cost # NEEDED FOR RAY DEMO

@contextmanager
def maybe_freeze_updates(enable):
    if enable:
        with freeze_collection([tf.GraphKeys.UPDATE_OPS]):
            yield
    else:
        yield

class TFBenchModel(Model):
    def _get_logits(self, image):
        ctx = get_current_tower_context()

        if args.use_fp16:
            image = tf.cast(image, tf.float16)
        with maybe_freeze_updates(ctx.index > 0):
            network = ConvNetBuilder(
                image, 3, True,
                use_tf_layers=True,
                data_format=self.data_format,
                dtype=tf.float16 if args.use_fp16 else tf.float32,
                variable_dtype=tf.float32)
            with custom_getter_scope(network.get_custom_getter()):
                dataset = lambda: 1
                dataset.name = 'imagenet'
                model_conf = model_config.get_model_config('resnet50', dataset)
                model_conf.set_batch_size(args.batch)
                model_conf.add_inference(network)
                return network.affine(1000, activation='linear', stddev=0.001)


class TensorpackModel(Model):
    """
    Implement the same model with tensorpack layers.
    """
    def _get_logits(self, image):
        # assert not args.use_fp16
        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                l = Conv2D('convshortcut', l, n_out, 1, strides=stride)
                l = BatchNorm('bnshortcut', l)
                return l
            else:
                return l

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            input = l
            l = Conv2D('conv1', l, ch_out, 1, strides=stride, activation=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, strides=1, activation=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1, activation=tf.identity)
            l = BatchNorm('bn', l)
            ret = l + shortcut(input, ch_in, ch_out * 4, stride)
            return tf.nn.relu(ret)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        defs = [3, 4, 6, 3]

        with argscope(Conv2D, use_bias=False,
                      kernel_initializer=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, strides=2)
                      .BatchNorm('bn0')
                      .tf.nn.relu()
                      .MaxPooling('pool0', 3, strides=2, padding='SAME')
                      .apply(layer, 'group0', bottleneck, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', bottleneck, 128, defs[1], 2)
                      .apply(layer, 'group2', bottleneck, 256, defs[2], 2)
                      .apply(layer, 'group3', bottleneck, 512, defs[3], 2)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000)())
        return logits


def get_data(mode):
    # get input
    input_shape = [args.batch, 224, 224, 3]
    label_shape = [args.batch]
    dataflow = FakeData(
        [input_shape, label_shape], 1000,
        random=False, dtype=[IMAGE_DTYPE_NUMPY, 'int32'])
    if mode == 'gpu':
        return DummyConstantInput([input_shape, label_shape])
    elif mode == 'cpu':
        def fn():
            # these copied from tensorflow/benchmarks
            with tf.device('/cpu:0'):
                if IMAGE_DTYPE == tf.float32:
                    images = tf.truncated_normal(
                        input_shape, dtype=IMAGE_DTYPE, stddev=0.1, name='synthetic_images')
                else:
                    images = tf.random_uniform(
                        input_shape, minval=0, maxval=255, dtype=tf.int32, name='synthetic_images')
                    images = tf.cast(images, IMAGE_DTYPE)
                labels = tf.random_uniform(
                    label_shape, minval=1, maxval=1000, dtype=tf.int32, name='synthetic_labels')
                # images = tf.contrib.framework.local_variable(images, name='images')
            return [images, labels]
        ret = TensorInput(fn)
        return StagingInput(ret, nr_stage=1)
    elif mode == 'python' or mode == 'python-queue':
        ret = QueueInput(
            dataflow,
            queue=tf.FIFOQueue(args.prefetch, [IMAGE_DTYPE, tf.int32]))
        return StagingInput(ret, nr_stage=1)
    elif mode == 'python-dataset':
        ds = TFDatasetInput.dataflow_to_dataset(dataflow, [IMAGE_DTYPE, tf.int32])
        ds = ds.repeat().prefetch(args.prefetch)
        ret = TFDatasetInput(ds)
        return StagingInput(ret, nr_stage=1)
    elif mode == 'zmq-serve':
        send_dataflow_zmq(dataflow, 'ipc://testpipe', hwm=args.prefetch, format='zmq_op')
        sys.exit()
    elif mode == 'zmq-consume':
        ret = ZMQInput(
            'ipc://testpipe', hwm=args.prefetch)
        return StagingInput(ret, nr_stage=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--model', choices=['tfbench', 'tensorpack'], default='tfbench')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--prefetch', type=int, default=150)
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--batch', type=int, default=64, help='per GPU batch size')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('--fake-location', help='the place to create fake data',
                        type=str, default='gpu',
                        choices=[
                            'cpu', 'gpu',
                            'python', 'python-queue', 'python-dataset',
                            'zmq-serve', 'zmq-consume'])
    parser.add_argument('--variable-update', help='variable update strategy',
                        type=str,
                        choices=['replicated', 'parameter_server', 'horovod'],
                        required=True)

    parser.add_argument('--ps-hosts')
    parser.add_argument('--worker-hosts')
    parser.add_argument('--job')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    sessconf = get_default_sess_config()
    sessconf.inter_op_parallelism_threads = 80 - 16
    if args.job:
        # distributed:
        cluster_spec = tf.train.ClusterSpec({
            'ps': args.ps_hosts.split(','),
            'worker': args.worker_hosts.split(',')
        })
        job = args.job.split(':')[0]
        if job == 'ps':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        task_index = int(args.job.split(':')[1])
        server = tf.train.Server(
            cluster_spec, job_name=job, task_index=task_index,
            config=sessconf)

    NR_GPU = get_nr_gpu()

    if args.job:
        trainer = {
            'replicated': lambda: DistributedTrainerReplicated(NR_GPU, server),
            'parameter_server': lambda: DistributedTrainerParameterServer(NR_GPU, server),
        }[args.variable_update]()
    else:
        if NR_GPU == 1:
            trainer = SimpleTrainer()
        else:
            trainer = {
                'replicated': lambda: SyncMultiGPUTrainerReplicated(
                    NR_GPU, average=False, mode='hierarchical' if NR_GPU >= 8 else 'cpu'),
                    # average=False is the actual configuration used by tfbench
                'horovod': lambda: HorovodTrainer(),
                'parameter_server': lambda: SyncMultiGPUTrainerParameterServer(NR_GPU, ps_device='cpu')
            }[args.variable_update]()

    M = TFBenchModel if args.model == 'tfbench' else TensorpackModel
    config = TrainConfig(
        data=get_data(args.fake_location),
        model=M(data_format=args.data_format),
        callbacks=[
            GPUUtilizationTracker(),
            # ModelSaver(checkpoint_dir='./tmpmodel'),  # it takes time
        ],
        extra_callbacks=[
            # MovingAverageSummary(),   # tensorflow/benchmarks does not do this
            ProgressBar(),  # nor this
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        session_config=sessconf if not args.job else None,
        steps_per_epoch=50,
        max_epoch=10,
    )


    # consistent with tensorflow/benchmarks
    trainer.COLOCATE_GRADIENTS_WITH_OPS = False
    launch_train_with_config(config, trainer)
