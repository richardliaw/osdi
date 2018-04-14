#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"

#include "plasma/client.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    LOG(INFO) << "INFO";
    LOG(ERROR) << "Logging ERROR";
    CHECK(0) << "Check";
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::Status::OK();
    });
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);

class MemcpyPlasmaOp : public OpKernel {
 public:
  explicit MemcpyPlasmaOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("plasma_store_socket_name",
                                             &plasma_store_socket_name_));
    OP_REQUIRES_OK(context, context->GetAttr("plasma_manager_socket_name",
                                             &plasma_manager_socket_name_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& plasma_object_id = context->input(1);

    // TODO(zongheng): copy into plasma object_id.
    plasma::PlasmaClient client;
    LOG(INFO) << "Connecting to Plasma";
    ARROW_CHECK_OK(client.Connect(plasma_store_socket_name_,
                                  plasma_manager_socket_name_,
                                  /*PLASMA_DEFAULT_RELEASE_DELAY=*/64));
    LOG(INFO) << "Connected to Plasma!";
    ARROW_CHECK_OK(client.Disconnect());

    return;
  }

 private:
  std::string plasma_store_socket_name_;
  std::string plasma_manager_socket_name_;
};

REGISTER_OP("MemcpyPlasma")
    .Input("input_tensor: float32")
    .Input("plasma_object_id: string")
    .Attr("plasma_store_socket_name: string")
    .Attr("plasma_manager_socket_name: string");

REGISTER_KERNEL_BUILDER(Name("MemcpyPlasma").Device(DEVICE_CPU),
                        MemcpyPlasmaOp);
