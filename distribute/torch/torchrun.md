# torchrun (Elastic Launch)

Superset of `torch.distributed.launch`.

torchrun提供了作为`torch.distributed.launch`的功能的超集，并具有以下附加功能：

1. 通过重新启动所有工作程序，可以优雅地处理工作程序故障。
2. Worker RANK和WORLD_SIZE是自动分配的。
3. 允许节点数量在最小和最大大小之间变化（弹性）

> 注释
> torchrun是setup.py中entry_points配置中声明的主模块torch.distributed.run的python控制台脚本。它相当于调用python-m torch.ddistributed.run。

## 从torch.distributed.slaunch过渡到torcrun

torchrun支持与torch.distributed.launch相同的参数，但--use env除外，后者现在已被弃用。要从torch.distributed.launch迁移到torcrun，请执行以下步骤：

1. 如果您的训练脚本已经在从`LOCAL_RANK`环境变量读取`local_rank`。然后只需省略`--use env`标志，例如：

`torch.distributed.launch`

`$ python -m torch.distributed.launch --use-env train_script.py`

`torchrun`

`$ torchrun train_script.py`

如果您的训练脚本从`--local-rank` cmd参数中读取了local rank。如以下代码片段所示，将训练脚本更改为从`LOCAL_RANK`环境变量读取：

`torch.distributed.launch`

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()

local_rank = args.local_rank
```

`torchrun`

```python
import os
local_rank = int(os.environ["LOCAL_RANK"])
```

上述更改足以从torch.distributed.slaunch迁移到torcrun。要利用torchrun的弹性、容错和错误报告等新功能，请参阅：

- 训练脚本，了解有关创作符合torchrun的训练脚本的更多信息。
- 本页的其余部分提供了更多关于torchrun功能的信息

### Usage

#### Single-node multi-worker

```bash
torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```


#### Stacked single-node multi-worker(堆叠单节点多工作者)

要在同一主机上运行单个节点、多个工作者的多个实例（单独的作业），我们需要确保每个实例（作业）都设置在不同的端口上，以避免端口冲突（或者更糟的是，两个作业合并为一个作业）。要做到这一点，您必须使用`--rdzv-backend=c10d`运行，并通过设置`--rdzv-endpoint=localhost:$PORT_k`指定不同的端口。对于`--nodes=1`，让`torcrun`自动选择一个可用的随机端口通常很方便，而不是为每次运行手动分配不同的端口。

```bash
torchrun
    --rdzv-backend=c10d
    --rdzv-endpoint=localhost:0
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```


#### Fault tolerant (fixed sized number of workers, no elasticity, tolerates 3 failures)[容错（固定数量的工人，无弹性，可容忍3次故障）]

```bash
torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

`HOST_NODE_ADDR`，格式为<HOST>[：<port>]（例如node1.example.com:29400），指定C10d会合后端应实例化和托管的节点和端口。它可以是训练集群中的任何节点，但理想情况下，您应该选择具有高带宽的节点。

> 注释
> 如果未指定端口号，HOST_NODE_ADDR默认为29400。

#### Elastic (`<span class="pre">min=1</span>`, `<span class="pre">max=4</span>`, tolerates up to 3 membership changes or failures)[弹性（最小值=1，最大值=4，最多可容忍3次成员身份更改或失败）]

```bash
torchrun
    --nnodes=1:4
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

`HOST_NODE_ADDR`，格式为<HOST>[：<port>]（例如node1.example.com:29400），指定C10d会合后端应实例化和托管的节点和端口。它可以是训练集群中的任何节点，但理想情况下，您应该选择具有高带宽的节点。

> 注释
> 如果未指定端口号，HOST_NODE_ADDR默认为29400。

### Note on rendezvous backend(关于会合后端的说明)

对于多节点训练，您需要指定：

1. `--rdzv-id`：唯一的作业id（由参与作业的所有节点共享）
2. `--rdzv-backend`：torch.distributed.aelastic.rendezvous.RendezvousHandler的一个实现。
3. `--rdzv-endpoint`：运行会合后端的端点；通常形式为host:port。

目前，开箱即用支持`c10d`（推荐）、`etcd-v2`和`etcd`（传统）会合后端。要使用`etcd-v2`或`etcd`，请设置一个启用`v2` api的etcd服务器（例如`--enable-v2`）。

> 警告
> `etcd-v2`和`etcd`会合使用etcd API v2。您必须在etcd服务器上启用v2 API。我们的测试使用etcd v3.4.3。

> 警告
> 对于etcd-based的交会，我们建议使用`etcd-v2`而不是`etcd`，这在功能上是等效的，但使用了修订的实现方式。`etcd`处于维护模式，将在未来版本中删除。

### Definitions(定义)

1. `Node`-一个物理实例或容器；映射到作业经理工作的单位。
2. `Worker`-分布式训练中的工人。
3. `WorkerGroup`-执行相同功能的一组工人（例如trainers）。
4. `LocalWorkerGroup`-在同一节点上运行的工作组中的工作人员的子集。
5. `RANK`-The rank of the worker within a worker group.
6. `WORLD_SIZE`-The total number of workers in a worker group.
7. `LOCAL_RANK`-The rank of the worker within a local worker group.
8. `LOCAL_WORLD_SIZE`-本地工作组的大小。
9. `rdzv_id`-用户定义的id，用于唯一标识作业的工作组。每个节点都使用此id作为特定工作组的成员加入。
10. `rdzv_backend`-集合的后端（例如c10d）。这通常是一个强一致的键值存储。
11. `rdzv_endpoint`-会合后端端点；通常形式为`host:port`<port>。

节点运行包含`LocalWorkerGroup`的`LOCAL_WORLD_SIZE`工作程序。作业中节点中所有`LocalWorkerGroup`的并集构成`WorkerGroup`。

### Environment Variables

The following environment variables are made available to you in your script:(脚本中提供了以下环境变量：)

1. `LOCAL_RANK` - The local rank.
2. `RANK` - The global rank.
3. `GROUP_RANK` - The rank of the worker group. A number between 0 and `max_nnodes`. When running a single worker group per node, this is the rank of the node.
4. `ROLE_RANK` - The rank of the worker across all the workers that have the same role. The role of the worker is specified in the `WorkerSpec`.
5. `LOCAL_WORLD_SIZE` - The local world size (e.g. number of workers running locally); equals to `--nproc-per-node` specified on `torchrun`.
6. `WORLD_SIZE` - The world size (total number of workers in the job).
7. `ROLE_WORLD_SIZE` - The total number of workers that was launched with the same role specified in `WorkerSpec`.
8. `MASTER_ADDR` - The FQDN of the host that is running worker with rank 0; used to initialize the Torch Distributed backend.
9. `MASTER_PORT` - The port on the `MASTER_ADDR` that can be used to host the C10d TCP store.
10. `TORCHELASTIC_RESTART_COUNT` - The number of worker group restarts so far.
11. `TORCHELASTIC_MAX_RESTARTS` - The configured maximum number of restarts.
12. `TORCHELASTIC_RUN_ID` - Equal to the rendezvous `run_id` (e.g. unique job id).
13. `PYTHON_EXEC` - System executable override. If provided, the python user script will use the value of `PYTHON_EXEC` as executable. The sys.executable is used by default.

### Deployment(部署)

1. （C10d后端不需要）启动会合后端服务器并获取端点（作为`--rdzv-endpoint`传递给启动器脚本）
2. 单节点多工作程序：在主机上启动启动器，启动创建和监视本地工作组的代理进程。
3. 多节点多工作者：在所有参与训练的节点上使用相同的参数启动启动器。

当使用作业/集群管理器时，多节点作业的入口点命令应该是这个启动器。

### Failure Modes

1. Worker failure: For a training job with `n` workers, if `k` workers fail all workers are stopped and restarted up to `max_restarts`.
2. Agent failure: An agent failure results in a local worker group failure. It is up to the job manager to fail the entire job (gang semantics) or attempt to replace the node. Both behaviors are supported by the agent.
3. Node failure: Same as agent failure.

### Membership Changes

1. Node departure (scale-down): The agent is notified of the departure, all existing workers are stopped, a new `WorkerGroup` is formed, and all workers are started with a new `RANK` and `WORLD_SIZE`.
2. Node arrival (scale-up): The new node is admitted to the job, all existing workers are stopped, a new `WorkerGroup` is formed, and all workers are started with a new `RANK` and `WORLD_SIZE`.

### Important Notices

1. This utility and multi-process distributed (single-node or multi-node) GPU training currently only achieves the best performance using the NCCL distributed backend. Thus NCCL backend is the recommended backend to use for GPU training.
2. The environment variables necessary to initialize a Torch process group are provided to you by this module, no need for you to pass `RANK` manually. To initialize a process group in your training script, simply run:

```python
import torch.distributed as dist
dist.init_process_group(backend="gloo|nccl")
```

3. In your training program, you can either use regular distributed functions or use `torch.nn.parallel.DistributedDataParallel` module. If your training program uses GPUs for training and you would like to use `torch.nn.parallel.DistributedDataParallel` module, here is how to configure it.

```python
local_rank = int(os.environ["LOCAL_RANK"])
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[local_rank],
                                                  output_device=local_rank)
```

Please ensure that `device_ids` argument is set to be the only GPU device id that your code will be operating on. This is generally the local rank of the process. In other words, the `device_ids` needs to be `[int(os.environ("LOCAL_RANK"))]`, and `output_device` needs to be `int(os.environ("LOCAL_RANK"))` in order to use this utility

4. On failures or membership changes ALL surviving workers are killed immediately. Make sure to checkpoint your progress. The frequency of checkpoints should depend on your job’s tolerance for lost work.
5. This module only supports homogeneous `LOCAL_WORLD_SIZE`. That is, it is assumed that all nodes run the same number of local workers (per role).
6. `RANK` is NOT stable. Between restarts, the local workers on a node can be assigned a different range of ranks than before. NEVER hard code any assumptions about the stable-ness of ranks or some correlation between `RANK` and `LOCAL_RANK`.
7. When using elasticity (`min_size!=max_size`) DO NOT hard code assumptions about `WORLD_SIZE` as the world size can change as nodes are allowed to leave and join.
8. It is recommended for your script to have the following structure:

```python
def main():
  load_checkpoint(checkpoint_path)
  initialize()
  train()

def train():
  for batch in iter(dataset):
    train_step(batch)

    if should_checkpoint:
      save_checkpoint(checkpoint_path)
```

9. (Recommended) On worker errors, this tool will summarize the details of the error (e.g. time, rank, host, pid, traceback, etc). On each node, the first error (by timestamp) is heuristically reported as the “Root Cause” error. To get tracebacks as part of this error summary print out, you must decorate your main entrypoint function in your training script as shown in the example below. If not decorated, then the summary will not include the traceback of the exception and will only contain the exitcode. For details on torchelastic error handling see: [https://pytorch.org/docs/stable/elastic/errors.html](https://pytorch.org/docs/stable/elastic/errors.html)

```python
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    # do train
    pass

if __name__ == "__main__":
    main()
```
