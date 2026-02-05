import torch

@torch.library.impl("aten::mul.Tensor", "npu")
def my_fallback(x, y):
    raise NotImplementedError("Fallback called")

if __name__ == "__main__":
    #torch.npu.register_fallback_op("aten::add.out", my_fallback)
    device = torch.device("npu:0")
    x = torch.ones(10, 10).to(device)
    y = torch.ones(10, 10).to(device)
    z = x * y
    print(z.cpu())