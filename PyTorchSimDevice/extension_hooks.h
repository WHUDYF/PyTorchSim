#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <ATen/core/Generator.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>

struct ExtensionPU1Hooks final : public at::PrivateUse1HooksInterface {
  ExtensionPU1Hooks() {}
  bool isBuilt() const;
  bool isAvailable() const;

  const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) const override;

  at::Generator getNewGenerator(c10::DeviceIndex device_index = -1) const override;

  at::Device getDeviceFromPtr(void* data) const override;

  bool isPinnedPtr(const void* data) const override;

  at::Allocator* getPinnedMemoryAllocator() const override;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;

  void resizePrivateUse1Bytes(const c10::Storage& /*storage*/, size_t /*newsize*/) const override;
};