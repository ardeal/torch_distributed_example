# torch_distributed_example

```python
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
```
1) command to run the upper code: 
    *  `CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 RANK=0 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=4 torch_distributed_example.py`
2) `torch.distributed.launch' will do a few things to run multiprocessing:
    + set 'local_rank' to args
    + launch the function `spmd_main` in torch_distributed_example.py `WORLD_SIZE` times
3) in `spmd_main` function, we could get `local_rank`, `CUDA_VISIBLE_DEVICES`, `WORLD_SIZE` and etc. According to those args, we could do what we want to do. 
4) Important: `local_rank` will change by `torch.distributed.launch` automatically. That is, `local_rank' we get in `spmd_main` function in different processing is differenet.
5) more reference:
    - https://github.com/pytorch/examples/tree/master/distributed/ddp
    - https://pytorch.org/docs/master/notes/ddp.html


