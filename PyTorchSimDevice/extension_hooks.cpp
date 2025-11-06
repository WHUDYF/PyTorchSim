#include "extension_hooks.h"

bool ExtensionPU1Hooks::isBuilt() const { return true; }
bool ExtensionPU1Hooks::isAvailable() const { return true; }

const at::Generator& ExtensionPU1Hooks::getDefaultGenerator(c10::DeviceIndex idx) const {
  if (idx < 0) idx = 0;
  static std::vector<at::Generator> gens;
  static std::mutex m;
  std::lock_guard<std::mutex> g(m);
  if (gens.size() <= (size_t)idx) gens.resize((size_t)idx + 1);
  if (!gens[idx].defined()) gens[idx] = at::GetGeneratorForPrivateuse1(idx);
  return gens[idx]; // 영속 객체 참조 반환
}

at::Generator ExtensionPU1Hooks::getNewGenerator(c10::DeviceIndex idx) const {
  if (idx < 0) idx = 0;
  return at::GetGeneratorForPrivateuse1(idx);
}

at::Device ExtensionPU1Hooks::getDeviceFromPtr(void* data) const {
  return at::Device(at::kPrivateUse1, 0); // MVP: 단일 디바이스 가정
}

bool ExtensionPU1Hooks::isPinnedPtr(const void* data) const {
  return false;
}

at::Allocator* ExtensionPU1Hooks::getPinnedMemoryAllocator() const {
  return at::getHostAllocator(at::kPrivateUse1);
}

bool ExtensionPU1Hooks::hasPrimaryContext(c10::DeviceIndex device_index) const { return true; }

void ExtensionPU1Hooks::resizePrivateUse1Bytes(const c10::Storage&, size_t) const {
  TORCH_CHECK(false, "resizePrivateUse1Bytes not implemented");
}

// REGISTER_EXTENSION_HOOKS(ExtensionPU1Hooks);

namespace {
struct AutoRegistrar {
  AutoRegistrar() {
    at::RegisterPrivateUse1HooksInterface(new ExtensionPU1Hooks());
  }
};
static AutoRegistrar _auto_registrar;
}
