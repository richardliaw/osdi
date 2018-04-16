
#include "plasma/client.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream.h"

using namespace tensorflow;

using ArrowStatus = arrow::Status;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// TODO(zongheng): CPU kernels' std::memcpy might be able to be sped up by
// parallelization.

// Put:  tf.Tensor -> plasma.
template <typename Device> class TensorToPlasmaOp : public AsyncOpKernel {
public:
  explicit TensorToPlasmaOp(OpKernelConstruction *context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("plasma_store_socket_name",
                                             &plasma_store_socket_name_));
    OP_REQUIRES_OK(context, context->GetAttr("plasma_manager_socket_name",
                                             &plasma_manager_socket_name_));
    VLOG(1) << "Connecting to Plasma...";
    mutex_lock lock(mu_);
    ARROW_CHECK_OK(client_.Connect(plasma_store_socket_name_,
                                   plasma_manager_socket_name_,
                                   PLASMA_DEFAULT_RELEASE_DELAY));
    VLOG(1) << "Connected!";
  }

  ~TensorToPlasmaOp() override {
    mutex_lock lock(mu_);
    ARROW_CHECK_OK(client_.Disconnect());
  }

  void ComputeAsync(OpKernelContext *context, DoneCallback done) override {
    const Tensor &input_tensor = context->input(0);
    const Tensor &plasma_object_id = context->input(1);
    CHECK(plasma_object_id.NumElements() == 1);
    const string &plasma_object_id_str =
        plasma_object_id.flat<std::string>()(0);

    VLOG(1) << "plasma_object_id_str: '" << plasma_object_id_str << "'";
    const plasma::ObjectID object_id =
        plasma::ObjectID::from_binary(plasma_object_id_str);

    const size_t input_bytes = input_tensor.TotalBytes();
    const size_t input_elems = input_tensor.NumElements();
    VLOG(1) << input_bytes << " " << input_elems;

    std::shared_ptr<Buffer> data_buffer;
    {
      mutex_lock lock(mu_);
      ARROW_CHECK_OK(client_.Create(object_id,
                                    static_cast<int64_t>(input_bytes),
                                    /*metadata=*/nullptr, 0, &data_buffer));
    }

    float *data = reinterpret_cast<float *>(data_buffer->mutable_data());

    auto wrapped_callback = [this, done, &object_id]() {
      {
        mutex_lock lock(mu_);
        ARROW_CHECK_OK(client_.Seal(object_id));
      }
      done();
    };

    if (std::is_same<Device, CPUDevice>::value) {
      std::memcpy(data, input_tensor.flat<float>().data(), input_bytes);
      wrapped_callback();
    } else {
      auto *stream = context->op_device_context()->stream();
      OP_REQUIRES_ASYNC(context, stream != nullptr,
                        errors::Internal("No GPU stream available."), done);
      perftools::gputools::DeviceMemoryBase wrapped_src(static_cast<void *>(
          const_cast<float *>(input_tensor.flat<float>().data())));
      // TODO(zongheng): do we need to somehow call HostMemoryRegister()?
      const bool success =
          stream
              ->ThenMemcpy(static_cast<void *>(data), wrapped_src,
                           static_cast<uint64>(input_bytes))
              .ok();
      OP_REQUIRES_ASYNC(context, success,
                        errors::Internal("D2H memcpy failed to be enqueued."),
                        done);
      context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
          stream, std::move(wrapped_callback));
    }
  }

private:
  std::string plasma_store_socket_name_;
  std::string plasma_manager_socket_name_;

  mutex mu_;
  plasma::PlasmaClient client_ GUARDED_BY(mu_);
};

// Get:  plasma -> tf.Tensor.
template <typename Device> class PlasmaToTensorOp : public AsyncOpKernel {
public:
  explicit PlasmaToTensorOp(OpKernelConstruction *context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("plasma_store_socket_name",
                                             &plasma_store_socket_name_));
    OP_REQUIRES_OK(context, context->GetAttr("plasma_manager_socket_name",
                                             &plasma_manager_socket_name_));
    VLOG(1) << "Connecting to Plasma...";
    mutex_lock lock(mu_);
    ARROW_CHECK_OK(client_.Connect(plasma_store_socket_name_,
                                   plasma_manager_socket_name_,
                                   PLASMA_DEFAULT_RELEASE_DELAY));
    VLOG(1) << "Connected!";
  }

  ~PlasmaToTensorOp() override {
    mutex_lock lock(mu_);
    ARROW_CHECK_OK(client_.Disconnect());
  }

  void ComputeAsync(OpKernelContext *context, DoneCallback done) override {
    const Tensor &plasma_object_id = context->input(0);
    CHECK(plasma_object_id.NumElements() == 1);
    const string &plasma_object_id_str =
        plasma_object_id.flat<std::string>()(0);

    VLOG(1) << "plasma_object_id_str: '" << plasma_object_id_str << "'";
    const plasma::ObjectID object_id =
        plasma::ObjectID::from_binary(plasma_object_id_str);

    plasma::ObjectBuffer object_buffer;
    {
      mutex_lock lock(mu_);
      ARROW_CHECK_OK(client_.Get(&object_id, /*num_objects=*/1,
                                 /*timeout_ms=*/-1, &object_buffer));
    }

    const int64_t size_in_bytes = object_buffer.data->size();
    TensorShape shape({size_in_bytes / sizeof(float)});
    VLOG(1) << "Output TensorShape: " << shape.DebugString();

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, shape, &output_tensor), done);

    if (std::is_same<Device, CPUDevice>::value) {
      std::memcpy(output_tensor->flat<float>().data(),
                  reinterpret_cast<const float *>(object_buffer.data->data()),
                  size_in_bytes);
      done();
    } else {
      auto *stream = context->op_device_context()->stream();
      OP_REQUIRES_ASYNC(context, stream != nullptr,
                        errors::Internal("No GPU stream available."), done);
      perftools::gputools::DeviceMemoryBase wrapped_dst(
          static_cast<void *>(output_tensor->flat<float>().data()));
      // TODO(zongheng): do we need to somehow call HostMemoryRegister()?
      const bool success = stream
                               ->ThenMemcpy(&wrapped_dst,
                                            reinterpret_cast<const void *>(
                                                object_buffer.data->data()),
                                            static_cast<uint64>(size_in_bytes))
                               .ok();
      OP_REQUIRES_ASYNC(context, success,
                        errors::Internal("D2H memcpy failed to be enqueued."),
                        done);
      context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
          stream, std::move(done));
    }
  }

private:
  std::string plasma_store_socket_name_;
  std::string plasma_manager_socket_name_;

  mutex mu_; // To guard "client_".
  plasma::PlasmaClient client_ GUARDED_BY(mu_);
};

REGISTER_OP("TensorToPlasma")
    .Input("input_tensor: float")
    .Input("plasma_object_id: string")
    .Attr("plasma_store_socket_name: string")
    .Attr("plasma_manager_socket_name: string");

REGISTER_KERNEL_BUILDER(Name("TensorToPlasma").Device(DEVICE_CPU),
                        TensorToPlasmaOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TensorToPlasma").Device(DEVICE_GPU),
                        TensorToPlasmaOp<GPUDevice>);

REGISTER_OP("PlasmaToTensor")
    .Input("plasma_object_id: string")
    .Output("tensor: float")
    .Attr("plasma_store_socket_name: string")
    .Attr("plasma_manager_socket_name: string");

REGISTER_KERNEL_BUILDER(Name("PlasmaToTensor").Device(DEVICE_CPU),
                        PlasmaToTensorOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("PlasmaToTensor").Device(DEVICE_GPU),
                        PlasmaToTensorOp<GPUDevice>);
