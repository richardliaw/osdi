#!/bin/bash
set -ex
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

module=memcpy_plasma_op

g++ -std=c++11 -shared ${module}.cc -o ${module}.so -fPIC \
    -undefined dynamic_lookup \
    ${TF_CFLAGS[@]} \
    -I/Users/zongheng/Dropbox/workspace/riselab/ray-py3//thirdparty/pkg/arrow/cpp/build/cpp-install/include \
    ${TF_LFLAGS[@]} \
    -L/Users/zongheng/Dropbox/workspace/riselab/ray-py3//thirdparty/pkg/arrow/cpp/build/cpp-install/lib -larrow -lplasma \
    -O2
