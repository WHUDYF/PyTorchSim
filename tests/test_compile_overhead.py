import os
import time
import sys
import torch
from torchvision.models import resnet18 as model1
import argparse
import shutil

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request, poisson_request_generator
CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')

if __name__ == "__main__":
    target_model1 = model1().eval()

    # Init scheduler
    for i in range(1):
        timestamp = time.time()  # 현재 타임스탬프 (초 단위)
        print(f"[{i}] Time Stamp: {timestamp:.6f}")  # 소수점 6자리까지 출력
        #try:
        #    shutil.rmtree("/tmp/torchinductor")
        #except FileNotFoundError:
        #    print("no cache")
        scheduler = Scheduler(num_request_queue=1, max_batch=4, engine_select=Scheduler.FIFO_ENGINE, togsim_config=f"{CONFIG_TORCHSIM_DIR}/configs/systolic_ws_128x128_c2_simple_noc_tpuv2.yml")
        # Register compiled model
        opt_model1 = torch.compile(target_model1.to(device=scheduler.execution_engine.module.custom_device(), memory_format=torch.channels_last), dynamic=False)
        SchedulerDNNModel.register_model("resnet18", opt_model1)

        # Generate time stamp
        for request_time in [0]*12:
            # Init input data
            model_input1 = torch.randn(1, 3, 224, 224)

            # Init request
            new_request1 = Request("resnet18", [model_input1], [], request_queue_idx=0)

            # Add request to scheduler
            print("[Reqest] Resnet18 request time: ", request_time, flush=True)
            scheduler.add_request(new_request1, request_time=request_time)

        # Run scheduler
        while not scheduler.is_finished():
            scheduler.schedule()

    print("Done", file=sys.stderr)