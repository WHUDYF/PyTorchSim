#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/MemoryFormat.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/CPUFallback.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace {
  bool g_amp_enabled = false;
  at::ScalarType g_amp_dtype = at::kFloat;
}

static at::ScalarType to_scalar_type(const py::object& dtype_obj) {
  py::module torch_mod = py::module::import("torch");
  if (dtype_obj.is(torch_mod.attr("bfloat16"))) return at::kBFloat16;
  if (dtype_obj.is(torch_mod.attr("float16")))  return at::kHalf;
  if (dtype_obj.is(torch_mod.attr("float32")))  return at::kFloat;
  if (dtype_obj.is(torch_mod.attr("float64")))  return at::kDouble;
  throw std::runtime_error("Unsupported dtype for extension_device AMP");
}

static py::object to_torch_dtype(at::ScalarType st) {
  py::module torch_mod = py::module::import("torch");
  switch (st) {
    case at::kBFloat16: return torch_mod.attr("bfloat16");
    case at::kHalf:     return torch_mod.attr("float16");
    case at::kFloat:    return torch_mod.attr("float32");
    case at::kDouble:   return torch_mod.attr("float64");
    default:
      throw std::runtime_error("Unsupported scalar type in get_autocast_dtype");
  }
}

static inline at::MemoryFormat fix_memory_format(c10::optional<at::MemoryFormat> mf_opt) {
    if (!mf_opt.has_value()) return at::MemoryFormat::Contiguous;

    auto mf = mf_opt.value();
    if (mf == at::MemoryFormat::Preserve) {
        return at::MemoryFormat::Contiguous;
    }
    return mf;
}

static uint64_t op_counter = 0;
static uint64_t last_saved_value = 0;

// register guard
namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);

}} // namespace at::detail

// basic dummy add function
at::Tensor custom_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  op_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

// basic dummy mul function
at::Tensor custom_mul_Tensor(const at::Tensor & self, const at::Tensor & other) {
  op_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

at::Tensor _reinterpret_tensor(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    int64_t offset_increment) {
  at::Tensor self_ = at::detail::make_tensor<c10::TensorImpl>(
      c10::Storage(self.storage()), self.key_set(), self.dtype());
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(self.storage_offset() + offset_increment);
  self_tmp_->set_sizes_and_strides(size, stride);
  return self_;
}

at::Tensor& zero_inplace_batching_rule(at::Tensor &self) {
  op_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return self;
}

const at::Tensor& custom_resize_(const at::Tensor& self, at::IntArrayRef size,
                          std::optional<at::MemoryFormat> optional_memory_format) {
  at::TensorImpl* tensor_impl = self.unsafeGetTensorImpl();
  tensor_impl->set_sizes_contiguous(size);
  const auto itemsize = tensor_impl->dtype().itemsize();
  const auto offset = tensor_impl->storage_offset();
  const auto storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize, offset);
  // Dummy device is using cpu allocator, so here just call cpu
  // function maybe_resize_storage_cpu in aten/src/ATen/native/Resize.h
  // to get a sufficient memory space.
  at::native::maybe_resize_storage_cpu(tensor_impl, storage_size);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != at::MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    tensor_impl->empty_tensor_restride(memory_format);
  }
  return self;
}

// basic dummy eq function: Only support CPU
at::Tensor custom_to_device(
    const at::Tensor & self,
    at::Device device,
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    c10::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(device.is_cpu() || device.type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  TORCH_CHECK(self.scalar_type() == dtype);
  TORCH_CHECK(self.is_contiguous());

  op_counter += 1;
  if (device.type() == at::DeviceType::CPU) {
    auto out = at::empty(self.sizes(), dtype, self.options().layout(),
                         device, false, memory_format);
    std::memcpy(out.mutable_data_ptr(), self.data_ptr(), self.nbytes());
    return out;
  } else {
    auto opts = self.options().device(device).dtype(dtype);
    auto out = at::empty(self.sizes(), opts);
    std::memcpy(out.mutable_data_ptr(), self.data_ptr(), self.nbytes());
    return out;
  }

  auto out = at::empty(self.sizes(), dtype, self.options().layout(), device, false, memory_format);
  memcpy(out.mutable_data_ptr(), self.mutable_data_ptr(), self.nbytes());
  // Since this custom device is just for testing, not bothering to implement kernels.
  return out;
}


// A dummy allocator for our custom device, that secretly uses the CPU
struct DummyCustomAllocator final : at::Allocator {
  DummyCustomAllocator() = default;
  at::DataPtr allocate(size_t nbytes) override {
    void* data = c10::alloc_cpu(nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    c10::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }
};

// Register our dummy allocator
static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "Dummy test only allows dummy device.");
  TORCH_CHECK(self.is_contiguous());

  op_counter += 1;

  switch (self.scalar_type()) {
    case c10::ScalarType::Float: {
      auto* data = self.mutable_data_ptr<float>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = value.toFloat();
      }
      break;
    }
    case c10::ScalarType::Double: {
      auto* data = self.mutable_data_ptr<double>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = value.toDouble();
      }
      break;
    }
    case c10::ScalarType::Half: {
      auto* data = self.mutable_data_ptr<at::Half>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = at::Half(value.toHalf());
      }
      break;
    }
    case c10::ScalarType::BFloat16: {
      auto* data = self.mutable_data_ptr<at::BFloat16>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = at::BFloat16(value.toBFloat16());
      }
      break;
    }
    case c10::ScalarType::Int: {
      auto* data = self.mutable_data_ptr<int>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = value.toInt();
      }
      break;
    }
    case c10::ScalarType::Long: {
      auto* data = self.mutable_data_ptr<int64_t>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = value.toLong();
      }
      break;
    }
    case c10::ScalarType::Short: {
      auto* data = self.mutable_data_ptr<int16_t>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = static_cast<int16_t>(value.toShort());
      }
      break;
    }
    case c10::ScalarType::Char: {
      auto* data = self.mutable_data_ptr<int8_t>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = static_cast<int8_t>(value.toChar());
      }
      break;
    }
    case c10::ScalarType::Byte: {
      auto* data = self.mutable_data_ptr<uint8_t>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = static_cast<uint8_t>(value.toByte());
      }
      break;
    }
    case c10::ScalarType::Bool: {
      auto* data = self.mutable_data_ptr<bool>();
      for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = value.toBool();
      }
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported scalar type: ", self.scalar_type());
  }
  return self;
}

at::Tensor unsafe_create_cpu_tensor_from_dummy_tensor(const at::Tensor& src) {
  // TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
  //             "Only support dummy device.");
  const auto& sizes_ = src.sizes();
  const auto& strides_ = src.strides();
  auto storage_offset_ = src.storage_offset();
  at::detail::check_size_nonnegative(sizes_);

  size_t size_bytes = at::detail::computeStorageNbytes(sizes_, strides_,
                                                       src.element_size(),
                                                       storage_offset_);

  at::DataPtr data_ptr =
    c10::InefficientStdFunctionContext::makeDataPtr(src.storage().mutable_data_ptr().get(),
                                                    [](void*){}, at::kCPU);

  c10::Storage storage{c10::Storage::use_byte_size_t{}, size_bytes, std::move(data_ptr),
    /*allocator=*/&global_custom_alloc, /*resizeable=*/false};

  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  at::Tensor tensor = at::detail::make_tensor<c10::TensorImpl>(
       std::move(storage), cpu_ks, src.dtype());

  c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  tensor_impl->set_sizes_and_strides(sizes_, strides_);
  tensor_impl->set_storage_offset(storage_offset_);
  return tensor;
}

// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(
      self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1,
      "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(
      dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1,
      "Dummy test only allows copy from cpu -> dummy device.");

  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  if (self.numel() != dst.numel()) {
    custom_resize_(dst, self.sizes(), c10::nullopt);
  }
  TORCH_CHECK(self.sizes() == dst.sizes());

  const bool same_dtype = (self.scalar_type() == dst.scalar_type());
  const bool both_contig = self.is_contiguous() && dst.is_contiguous();

  // 1) fast path
  if (same_dtype && both_contig) {
    std::memcpy(dst.mutable_data_ptr(),
                self.data_ptr(),
                dst.storage().nbytes());
    return dst;
  }

  // 2) slow path
  at::Tensor cpu_self = unsafe_create_cpu_tensor_from_dummy_tensor(self);
  at::Tensor cpu_dst  = unsafe_create_cpu_tensor_from_dummy_tensor(dst);
  if (!same_dtype) {
    cpu_self = cpu_self.to(cpu_dst.scalar_type(), /*non_blocking=*/false, /*copy=*/true);
  }
  cpu_dst.copy_(cpu_self);
  return dst;
}

at::Tensor custom__copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  return custom__copy_from(self, dst, false);
}

at::Tensor& custom_abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::abs_out(self, out);
}

at::Tensor custom_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  op_counter += 1;
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return  at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, dtype);
}

at::Tensor custom_empty(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt, c10::optional<c10::MemoryFormat> optional_memory_format) {
  op_counter += 1;

  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return  at::detail::empty_generic(size, &global_custom_alloc, private_use_ks, dtype, fix_memory_format(optional_memory_format));
}

at::Tensor& custom_arange_start_out_impl(
    const c10::Scalar& start,
    const c10::Scalar& end,
    const c10::Scalar& step,
    at::Tensor& out) {
  double s = start.toDouble();
  double e = end.toDouble();
  double st = step.toDouble();
  TORCH_CHECK(st != 0.0, "step must be nonzero");

  int64_t length = 0;
  if (st > 0) {
    if (e > s) length = static_cast<int64_t>(std::ceil((e - s) / st));
  } else {
    if (e < s) length = static_cast<int64_t>(std::ceil((e - s) / st));
  }

  // Resize out tensor
  custom_resize_(out, {length}, c10::nullopt);

  if (out.scalar_type() == at::kFloat || out.scalar_type() == at::kDouble) {
    double* data = out.mutable_data_ptr<double>();
    for (int64_t i = 0; i < length; i++) {
      data[i] = s + i * st;
    }
  } else if (out.scalar_type() == at::kLong) {
    int64_t* data = out.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < length; i++) {
      data[i] = static_cast<int64_t>(s + i * st);
    }
  } else {
    TORCH_CHECK(false, "Unsupported dtype for arange on dummy device");
  }

  return out;
}

static at::Tensor custom_to_dtype_impl(const at::Tensor& self,
                                       c10::ScalarType dtype,
                                       bool non_blocking, bool copy,
                                       c10::optional<c10::MemoryFormat> memory_format) {
  return at::native::to(self, dtype, non_blocking, copy, memory_format);
}

at::Tensor custom_zeros_like(
    const at::Tensor& input,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt)
{
  // dtype / layout / device fallback
  auto dtype   = dtype_opt.value_or(input.scalar_type());
  auto layout  = layout_opt.value_or(input.layout());
  auto device  = device_opt.value_or(input.device());
  auto memfmt  = memory_format_opt.value_or(c10::MemoryFormat::Contiguous);

  TORCH_CHECK(
      device.type() == c10::DeviceType::PrivateUse1,
      "custom_zeros_like: device must be PrivateUse1");

  at::Tensor out = custom_empty(
      input.sizes(),
      dtype,
      layout,
      device,
      pin_memory_opt,
      memfmt
  );
  size_t nbytes = out.numel() * out.element_size();
  void* ptr = out.mutable_data_ptr();

  TORCH_CHECK(ptr != nullptr,
      "custom_zeros_like: out.mutable_data_ptr() returned NULL");
  std::memset(ptr, 0, nbytes);
  return out;
}

at::Tensor& custom_zero_impl(at::Tensor& self)
{
    TORCH_CHECK(
        self.device().type() == c10::DeviceType::PrivateUse1,
        "custom_zero_: expected a PrivateUse1 device tensor");

    if (self.numel() == 0) {
        return self;
    }

    void* data = self.mutable_data_ptr();
    TORCH_CHECK(data != nullptr,
        "custom_zero_: self.mutable_data_ptr() returned NULL "
        "(storage was not allocated)");

    size_t nbytes = self.numel() * self.element_size();
    std::memset(data, 0, nbytes);

    return self;
}

// With TORCH_LIBRARY_IMPL, you can register custom kernels for your backend.
// For open registration, we're registering all of our kernels to the PrivateUse1 dispatch key.
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("to.Device",             &custom_to_device);
  m.impl("to.dtype",              &custom_to_dtype_impl);
  m.impl("fill_.Scalar",          &custom_fill__scalar);
  m.impl("_copy_from",            &custom__copy_from);
  m.impl("_copy_from_and_resize", &custom__copy_from_and_resize);
  m.impl("empty_strided",         &custom_empty_strided);
  m.impl("empty.memory_format",   &custom_empty);
  m.impl("as_strided",            at::native::as_strided_tensorimpl);
  m.impl("view",                  at::native::view);
  m.impl("arange.start_out",      &custom_arange_start_out_impl);
  m.impl("zeros_like",            &custom_zeros_like);
  m.impl("zero_",                 &custom_zero_impl);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("to.dtype", &custom_to_dtype_impl);
}

TORCH_LIBRARY_FRAGMENT(aten, m) {
  m.def(
    "_reinterpret_tensor(Tensor self, int[] size, int[] stride, int offset_increment=0) -> Tensor",
    torch::dispatch(c10::DispatchKey::AutogradPrivateUse1, _reinterpret_tensor),
    {at::Tag::pt2_compliant_tag}
  );
}

void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("abs", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("abs.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("abs_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("absolute", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("absolute.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("absolute_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("add.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("add.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("add.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("add_.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("add_.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("cat", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("cat.names", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("cat.names_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("cat.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("div.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("div.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("div.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("div_.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("div_.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("eq.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("eq.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("eq.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("eq.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("equal", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("erf", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("erf.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("erf_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("erfc", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("erfc.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("erfc_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("exp", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("exp.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("ge.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("ge.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("ge.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("ge.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("gt.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("gt.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("gt.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("gt.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("le.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("le.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("le.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("le.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("lt.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("lt.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("lt.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("lt.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("ne.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("ne.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("ne.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("ne.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("logical_and", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_and.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_and_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_not", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_not.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_not_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_or", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_or.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_or_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_xor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_xor.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("logical_xor_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("neg", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("neg.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("neg_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("mul.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("mul.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("mul_.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("pow.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("pow.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("pow.Tensor_Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("pow.Tensor_Scalar_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("pow.Tensor_Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("pow.Tensor_Tensor_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("pow_.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("pow_.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("sub.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sub.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sub.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sub_.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sub_.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("sum", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sum.DimnameList_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sum.IntList_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sum.dim_DimnameList", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sum.dim_IntList", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("resize_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("resize_as_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  // Foreach ops
  m.impl("_foreach_add.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_foreach_add_.Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_foreach_add_.ScalarList", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_foreach_add.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_foreach_add_.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  // Indexed
  m.impl("index_add.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index_add_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index_copy.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index_copy_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index_fill.int_Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index_fill.int_Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index_fill.int_Scalar_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index_fill.int_Tensor_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index_fill_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("tril", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("tril_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("triu", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("triu_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("triu_indices", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("nll_loss2d_forward", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("nll_loss2d_backward", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("nll_loss_backward", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("nll_loss_forward", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("scatter.src_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("scatter.value_out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("index_put.Default", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("mm.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sigmoid.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("gather.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("silu.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());

  m.impl("all.all_out",                   torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_local_scalar_dense",           torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_log_softmax",                  torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_log_softmax_backward_data",    torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("mse_loss.out",                  torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_native_multi_head_attention",  torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("where.self",                    torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("min",                           torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("max",                           torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index_select",                  torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("nonzero",                       torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

bool custom_op_called() {
  bool called = false;
  if (op_counter > last_saved_value) {
    called = true;
    last_saved_value = op_counter;
  }
  return called;
}

class PrivateGeneratorImpl : public at::CPUGeneratorImpl {
public:
  PrivateGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~PrivateGeneratorImpl() override = default;
};

// this is used to register generator
at::Generator make_generator_privateuse1(c10::DeviceIndex device_index) {
  return at::make_generator<PrivateGeneratorImpl>(device_index);
}

void register_generator() {
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("custom_device", &get_custom_device, "get custom device object");
  m.def("custom_op_called", &custom_op_called, "check if our custom function was called");
  m.def("register_generator", &register_generator, "register generator for custom device");
  m.def("is_autocast_enabled", []() -> bool { return g_amp_enabled;});
  m.def("set_autocast_enabled", [](bool flag) -> void {g_amp_enabled = flag;});
  m.def("get_autocast_dtype", []() -> py::object { return to_torch_dtype(g_amp_dtype); });
  m.def("set_autocast_dtype", [](py::object dtype_obj) -> void {
    auto st = to_scalar_type(dtype_obj);
    g_amp_dtype = st;
  });
  m.def("get_amp_supported_dtype", []() -> py::list {
    py::module torch_mod = py::module::import("torch");
    py::list lst;
    lst.append(torch_mod.attr("float16"));
    lst.append(torch_mod.attr("float32"));
    return lst;
  });
}