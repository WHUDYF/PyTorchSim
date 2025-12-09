import os
import sys
import torch
import argparse
sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request
from test_stonne import sparse_matmul

def custom_matmul(a, b):
    return torch.matmul(a, b)
torch.manual_seed(0)
CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--M", type=int, default=128, help="Batch size")
    parser.add_argument("--N", type=int, default=128, help="Input layer size")
    parser.add_argument("--K", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Output layer size")
    parser.add_argument("--config", type=str, default="stonne_big_c1_simple_noc.yml", help="Output layer size")
    parser.add_argument("--mode", type=int, default=0, help="Output layer size")
    args = parser.parse_args()

    M = args.M
    N = args.N
    K = args.K
    sparsity = args.sparsity
    mode = args.mode
    config_path = f"{CONFIG_TORCHSIM_DIR}/configs/{args.config}"

    print("M: ", M)
    print("N: ", N)
    print("K: ", K)
    print("sparsity: ", sparsity)

    with torch.no_grad():
        # Init scheduler
        scheduler = Scheduler(num_request_queue=2, engine_select=Scheduler.FIFO_ENGINE,
                            togsim_config=config_path)

        # Register compiled model
        opt_model1 = torch.compile(custom_matmul)
        opt_model2 = torch.compile(sparse_matmul)
        SchedulerDNNModel.register_model("matmul", opt_model1)
        SchedulerDNNModel.register_model("spmm", opt_model2)

        # Init input data
        for i in range(1):
            dense_input1 = torch.randn(M, K)
            dense_input2 = torch.randn(K, N)

            sparse_input1 = torch.randn(128, 128)
            sparse_input2 = torch.randn(128, 128)
            mask1 = torch.rand(sparse_input1.shape) > sparsity
            mask2 = torch.rand(sparse_input2.shape) > sparsity

            sparse_input1 = sparse_input1 * mask1
            sparse_input2 = sparse_input2 * mask2

            # Init request
            if mode == 0:
                new_request1 = Request("spmm", [sparse_input1, sparse_input2], [], request_queue_idx=0)
                scheduler.add_request(new_request1, request_time=0)
            elif mode == 1:
                new_request2 = Request("matmul", [dense_input1, dense_input2], [], request_queue_idx=0)
                scheduler.add_request(new_request2, request_time=0)
            elif mode == 2:
                new_request1 = Request("spmm", [sparse_input1, sparse_input2], [], request_queue_idx=0)
                new_request2 = Request("matmul", [dense_input1, dense_input2], [], request_queue_idx=1)

                # Add request to scheduler
                scheduler.add_request(new_request1, request_time=0)
                scheduler.add_request(new_request2, request_time=0)

        # Run scheduler
        while not scheduler.is_finished():
            scheduler.schedule()