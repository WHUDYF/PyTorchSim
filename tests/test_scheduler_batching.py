import os
import sys
import torch
from torchvision.models import resnet18 as model1
import argparse

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request, poisson_request_generator
CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisson Request Generator (ms)")
    parser.add_argument("lambda_requests", nargs="?", type=int, help="Average requests per second (λ)", default=2000)
    parser.add_argument("max_time", nargs="?", type=int, help="Maximum simulation time in milliseconds", default=30)

    args = parser.parse_args()
    target_model1 = model1().eval()

    # Init scheduler
    scheduler = Scheduler(num_request_queue=1, max_batch=32, engine_select=Scheduler.FIFO_ENGINE, togsim_config=f"{CONFIG_TORCHSIM_DIR}/configs/systolic_ws_128x128_c2_simple_noc_tpuv2.yml")
    # Register compiled model
    opt_model1 = torch.compile(target_model1.to(device=scheduler.execution_engine.module.custom_device(), memory_format=torch.channels_last), dynamic=False)
    SchedulerDNNModel.register_model("resnet18", opt_model1)

    # Generate time stamp
    for request_time in poisson_request_generator(args.lambda_requests, args.max_time):
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