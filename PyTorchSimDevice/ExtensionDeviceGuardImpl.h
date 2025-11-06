#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/Stream.h>
#include <c10/core/Event.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>

namespace c10::extension_device::impl {

struct ExtensionDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::PrivateUse1; // ✅ your backend type

  ExtensionDeviceGuardImpl() = default;

  explicit ExtensionDeviceGuardImpl(DeviceType t) {
    TORCH_CHECK(
        t == static_type,
        "ExtensionDeviceGuardImpl initialized with non-extension_device DeviceType: ",
        t);
  }

  // --------------------------------------------------------------------------
  // 기본적인 device guard (CPU처럼 동작)
  // --------------------------------------------------------------------------
  DeviceType type() const override {
    return static_type;
  }

  Device exchangeDevice(Device d) const override {
    TORCH_CHECK(d.type() == static_type, "Expected extension_device but got ", d);
    return d; // nothing to exchange, CPU-like
  }

  Device getDevice() const override {
    return Device(static_type, 0);
  }

  void setDevice(Device d) const override {
    TORCH_CHECK(d.type() == static_type, "Expected extension_device but got ", d);
  }

  void uncheckedSetDevice(Device d) const noexcept override {}

  DeviceIndex deviceCount() const noexcept override {
    return 1; // pretend single device
  }

  // --------------------------------------------------------------------------
  // Stream handling (동기식이므로 기본 stream만 사용)
  // --------------------------------------------------------------------------
  Stream getStream(Device d) const override {
    return Stream(Stream::DEFAULT, d);
  }

  Stream getNewStream(Device d, int priority = 0) const override {
    return Stream(Stream::DEFAULT, d);
  }

  Stream getStreamFromGlobalPool(Device d, bool = false) const override {
    return Stream(Stream::DEFAULT, d);
  }

  Stream exchangeStream(Stream s) const override {
    return s;
  }

  bool queryStream(const Stream& stream) const override {
    (void)stream;
    return true;
  }

  void synchronizeStream(const Stream& stream) const override {
    (void)stream;
  }

  void synchronizeDevice(DeviceIndex device_index) const override {
    (void)device_index;
  }

  // --------------------------------------------------------------------------
  // Event handling (전부 no-op)
  // --------------------------------------------------------------------------
  void destroyEvent(void* event, const DeviceIndex device_index) const noexcept override {
    (void)event;
    (void)device_index;
  }

  void record(void** event, const Stream& stream, const DeviceIndex device_index, const EventFlag flag) const override {
    (void)event;
    (void)stream;
    (void)device_index;
    (void)flag;
  }

  void block(void* event, const Stream& stream) const override {
    (void)event;
    (void)stream;
  }

  bool queryEvent(void* event) const override {
    (void)event;
    return true;
  }

  void synchronizeEvent(void* event) const override {
    (void)event;
  }

  double elapsedTime(void* start_event, void* end_event, const DeviceIndex device_index) const override {
    (void)start_event;
    (void)end_event;
    (void)device_index;
    return 0.0;
  }

  // --------------------------------------------------------------------------
  // Misc (allocator integration)
  // --------------------------------------------------------------------------
  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream) const override {
    (void)data_ptr;
    (void)stream;
  }
};

} // namespace c10::extension_device::impl
