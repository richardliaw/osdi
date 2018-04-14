#!/bin/bash
set -ex
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
RAY_ROOT="$(python -c 'import ray; print(ray.__path__[0])')/../../"

module=memcpy_plasma_op

if [ "$(uname)" == "Darwin" ]; then
    TF_CFLAGS="-undefined dynamic_lookup ${TF_CFLAGS}"
fi

# `pkg-config --cflags --libs plasma` \
g++ -std=c++11 -g -shared ${module}.cc -o ${module}.so \
    -I$RAY_ROOT/thirdparty/pkg/arrow/cpp/build/cpp-install/include \
    -L$RAY_ROOT/thirdparty/pkg/arrow/cpp/build/cpp-install/lib -lplasma \
    -fPIC \
    ${TF_CFLAGS[@]} \
    ${TF_LFLAGS[@]} \
    -O2

# g++ -std=c++11 -shared ${module}.cc -o ${module}.so -fPIC \
#     ${TF_CFLAGS[@]} \
#     `pkg-config --cflags plasma` \
#     ${TF_LFLAGS[@]} \
#     `pkg-config --libs plasma` \
#     -O2

# -L${RAY_ROOT}/thirdparty/pkg/arrow/cpp/build/cpp-install/lib -larrow \

# LD_LIBRARY_PATH=$RAY_ROOT/thirdparty/pkg/arrow/cpp/build/cpp-install/lib python memcpy_plasma_op_test.py
