import argparse
import os
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
 
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda:{}'.format(args.local_rank))
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # ####################################################################
    # you could now push your model(DL network) to gpu and do training now
    # #####################################################################

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This is passed in via launch.py, This needs to be explicitly passed in
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    spmd_main(args.local_world_size, args.local_rank)
 
