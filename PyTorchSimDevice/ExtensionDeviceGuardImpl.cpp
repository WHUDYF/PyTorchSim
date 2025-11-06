#include "ExtensionDeviceGuardImpl.h"
#include <c10/core/impl/DeviceGuardImplRegistry.h>

namespace c10::extension_device::impl {

C10_REGISTER_GUARD_IMPL(extension_device, ExtensionDeviceGuardImpl);

} // namespace c10::extension_device::impl
