#!/bin/bash

# Requirements on Ubuntu/Linux:
# Use
#    gcc / g++ < 5
# to compile Ray ("pip install -e ." from source) and the ops.
# See, e.g. https://askubuntu.com/questions/724872/downgrade-gcc-from-5-2-1-to-4-9-ubuntu-15-10
# This is to avoid weird CXX11 ABI issues.

set -ex
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

module=memcpy_plasma_op

if [ "$(uname)" == "Darwin" ]; then
    TF_CFLAGS="-undefined dynamic_lookup ${TF_CFLAGS}"
fi

NDEBUG=""
NDEBUG="-DNDEBUG"

# nvcc -std=c++11 -c -o ${module}.cu.o ${module}_gpu.cu.cc \
#      ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

RAY_ROOT="$(python -c 'import ray; print(ray.__path__[0])')/../../"
# For some reason "-L<plasma> -lplasma ... -L<tf> -l<tf>" does not work; however
# "-L<plasma lib> ... -L <tf> -l<tf> -l<plasma>" does work.
# ${module}.cu.o \
g++ -std=c++11 -g -shared ${module}.cc -o ${module}.so \
    ${NDEBUG}\
    -I$RAY_ROOT/thirdparty/pkg/arrow/cpp/build/cpp-install/include \
    -L$RAY_ROOT/thirdparty/pkg/arrow/cpp/build/cpp-install/lib \
    -fPIC \
    ${TF_CFLAGS[@]} \
    -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart \
    ${TF_LFLAGS[@]} -lplasma -larrow \
    -O2

# Proves to work with gcc < 5 on Ubuntu:
# g++ -std=c++11 -shared ${module}.cc -o ${module}.so \
#     -fPIC \
#     ${TF_CFLAGS[@]} \
#     -I$RAY_ROOT/thirdparty/pkg/arrow/cpp/build/cpp-install/include \
#     -L$RAY_ROOT/thirdparty/pkg/arrow/cpp/build/cpp-install/lib \
#     ${TF_LFLAGS[@]} \
#     -l:libplasma.a   -l:libarrow.a \
#     -O2

#####

# g++ -std=c++11 -shared ${module}.cc -o ${module}.so -fPIC \
#     ${TF_CFLAGS[@]} \
#     `pkg-config --cflags plasma` \
#     ${TF_LFLAGS[@]} \
#     `pkg-config --libs plasma` \
#     -O2

# -L${RAY_ROOT}/thirdparty/pkg/arrow/cpp/build/cpp-install/lib -larrow \

# LD_LIBRARY_PATH=$RAY_ROOT/thirdparty/pkg/arrow/cpp/build/cpp-install/lib python memcpy_plasma_op_test.py
