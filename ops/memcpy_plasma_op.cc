
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

static plasma::PlasmaClient client_;
static bool connected_ = false;
static mutex mu_;

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
    mutex_lock lock(mu_);
    if (!connected_) {
      VLOG(1) << "Connecting to Plasma...";
      ARROW_CHECK_OK(client_.Connect(plasma_store_socket_name_,
                                     plasma_manager_socket_name_,
                                     PLASMA_DEFAULT_RELEASE_DELAY));
      VLOG(1) << "Connected!";
      connected_ = true;
    }
  }

  ~TensorToPlasmaOp() override {
    mutex_lock lock(mu_);
    ARROW_CHECK_OK(client_.Disconnect());
  }

  void ComputeAsync(OpKernelContext *context, DoneCallback done) override {
    const int num_inputs = context->num_inputs();
    OP_REQUIRES_ASYNC(
        context, num_inputs >= 2,
        errors::InvalidArgument(
            "Input should have at least 1 tensor and 1 object_id"),
        done);
    const int num_tensors = num_inputs - 1;

    std::vector<size_t> offsets;
    offsets.reserve(num_tensors + 1);
    offsets.push_back(0);
    size_t total_bytes = 0;
    for (int i = 0; i < num_tensors; ++i) {
      const size_t s = context->input(i).TotalBytes();
      total_bytes += s;
      offsets.push_back(total_bytes);
    }

    // const Tensor &input_tensor = context->input(0);
    const Tensor &plasma_object_id = context->input(num_inputs - 1);
    CHECK(plasma_object_id.NumElements() == 1);
    const string &plasma_object_id_str =
        plasma_object_id.flat<std::string>()(0);
    VLOG(1) << "plasma_object_id_str: '" << plasma_object_id_str << "'";
    const plasma::ObjectID object_id =
        plasma::ObjectID::from_binary(plasma_object_id_str);

    // const size_t input_bytes = input_tensor.TotalBytes();
    // const size_t input_elems = input_tensor.NumElements();
    // VLOG(1) << input_bytes << " " << input_elems;

    // LOG(INFO) << "Creating Plasma buffer of size " << total_bytes;
    std::shared_ptr<Buffer> data_buffer;
    {
      mutex_lock lock(mu_);
      ARROW_CHECK_OK(client_.Create(object_id,
                                    static_cast<int64_t>(total_bytes),
                                    /*metadata=*/nullptr, 0, &data_buffer));
    }

    float *data = reinterpret_cast<float *>(data_buffer->mutable_data());

    auto wrapped_callback = [this, done, data_buffer, object_id]() {
      {
        mutex_lock lock(mu_);
        ARROW_CHECK_OK(client_.Seal(object_id));
      }

      // const float *plasma_data =
      //     reinterpret_cast<const float *>(data_buffer->data());
      // LOG(INFO) << "After Seal, printing what's in plasma buffer: ";
      // for (int i = 0; i < 6; ++i) {
      //   LOG(INFO) << plasma_data[i];
      // }

      done();
    };

    if (std::is_same<Device, CPUDevice>::value) {
      CHECK(num_tensors == 1)
          << "To be extended to >1 case for CPU; run with GPU.";
      std::memcpy(data, context->input(0).flat<float>().data(), total_bytes);
      wrapped_callback();
    } else {

      // Launch 1 memcpy per Tensor.
      auto *stream = context->op_device_context()->stream();
      OP_REQUIRES_ASYNC(context, stream != nullptr,
                        errors::Internal("No GPU stream available."), done);

      for (int i = 0; i < num_tensors; ++i) {
        const auto &input_tensor = context->input(i);
        // LOG(INFO) << "Memcpy ( tensor " << i
        //           << ", src, size=" << offsets[i + 1] - offsets[i]
        //           << "; num_elems " << input_tensor.NumElements();
        // LOG(INFO) << "offsets[i] = " << offsets[i];
        float *input_buffer =
            const_cast<float *>(input_tensor.flat<float>().data());
        // CHECK(input_buffer != nullptr);
        // for (int j = 0; j < input_tensor.NumElements(); ++j)
        //   LOG(INFO) << input_buffer[j];

        perftools::gputools::DeviceMemoryBase wrapped_src(
            static_cast<void *>(input_buffer));
        // TODO(zongheng): do we need to somehow call HostMemoryRegister()?
        const bool success =
            stream
          ->ThenMemcpy(static_cast<void *>(data + offsets[i] / sizeof(float)),
                             wrapped_src,
                             static_cast<uint64>(offsets[i + 1] - offsets[i]))
                .ok();
        OP_REQUIRES_ASYNC(context, success,
                          errors::Internal("D2H memcpy failed to be enqueued."),
                          done);
      }

      context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
          stream, wrapped_callback);
    }
  }

private:
  std::string plasma_store_socket_name_;
  std::string plasma_manager_socket_name_;

  // mutex mu_;
  // plasma::PlasmaClient client_ GUARDED_BY(mu_);
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
    mutex_lock lock(mu_);
    if (!connected_) {
      VLOG(1) << "Connecting to Plasma...";
      ARROW_CHECK_OK(client_.Connect(plasma_store_socket_name_,
                                     plasma_manager_socket_name_,
                                     PLASMA_DEFAULT_RELEASE_DELAY));
      VLOG(1) << "Connected!";
      connected_ = true;
    }
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
      // ARROW_CHECK_OK(client_.Get({object_id},
      //                            /*timeout_ms=*/-1, &object_buffer));
    }

    const int64_t size_in_bytes = object_buffer.data->size();
    TensorShape shape({size_in_bytes / sizeof(float)});
    // LOG(INFO) << "Output TensorShape: " << shape.DebugString();
    // LOG(INFO) << "size_in_bytes of the plasma object: " << size_in_bytes;

    const float *plasma_data =
        reinterpret_cast<const float *>(object_buffer.data->data());
    // for (int i = 0; i < size_in_bytes / sizeof(float); ++i) {
    //   LOG(INFO) << plasma_data[i];
    // }

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

  // mutex mu_;  // To guard "client_".
  // plasma::PlasmaClient client_ GUARDED_BY(mu_);
};

REGISTER_OP("TensorToPlasma")
    .Input("input_tensor: dtypes")
    .Input("plasma_object_id: string")
    .Attr("dtypes: list(type)")
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
