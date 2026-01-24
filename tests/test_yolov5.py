import torch
import torch._dynamo
import torch.utils.cpp_extension

import argparse
import datetime

import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms

import os
import shutil



def run_yolo(batch, config):
    from Scheduler.scheduler import PyTorchSimRunner
    device = PyTorchSimRunner.setup_device().custom_device()

    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 128
    
    model = torch.hub.load("ultralytics/yolov5", "yolov5s").cpu().eval()
    url = "https://ultralytics.com/images/zidane.jpg"
    
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    imgsz = 64    # 이미지 사이즈 줄여서 시뮬레이터 체크 가속
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])
    
    x = transform(img).unsqueeze(0)   # [1, 3, H, W]
    x = x.to(device)
    

    model.to(device)
    x = x.to(device)
    
    # Compile and run the model with PyTorchSim
    compiled_model = torch.compile(dynamic=False)(model)
    y = compiled_model(x)
    print("Yolo Simulation Done")


if __name__ == "__main__":
    import sys

    base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
    config = os.environ.get(
        "TORCHSIM_CONFIG",
        default=f"{base_dir}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml",
    )
    config_prefix = config.split("/")[-1].split(".")[0][
        9:
    ]  # extract config name from config path
    sys.path.append(base_dir)
    args = argparse.ArgumentParser()
    args.add_argument("--batch", type=int, default=1)
    args.add_argument("--dump_path", type=str, default="results")
    args = args.parse_args()
    batch = args.batch
    result_path = os.path.join(
        base_dir,
        args.dump_path,
        config_prefix,
        f"yolo5s_{batch}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    
    # setting environment variables
    os.environ["TORCHSIM_LOG_PATH"] = result_path
    os.environ["TORCHSIM_USE_TIMING_POOLING"] = "1"
    
    # only timing simulation
    os.environ["TORCHSIM_VALIDATION_MODE"] = "0"
    if "pytorchsim_functional_mode" in os.environ:
        del os.environ["pytorchsim_functional_mode"]

    # Clear extension/inductor caches to force rebuilds
    shutil.rmtree("/tmp/torchinductor_root", ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.cache/torch_extensions/py311_cu126/npu"), ignore_errors=True)

    run_yolo(batch, config)
