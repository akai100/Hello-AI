PyTorch 提供了两种主要的分布式训练方式：

+ Data Parallel：适用于单机多 GPU 的训练。

+ Distributed Data Parallel (DDP)：适用于多机多 GPU 的训练。

## Data Parallel

### 使用 torch.nn.DataParallel

DataParallel 是 PyTorch 中用于单机多 GPU 训练的方式。它通过自动分配数据到多个 GPU 上，并在每个 GPU 上计算梯度，最后将梯度汇总到主 GPU。


## Distributed Data Parallel (DDP)


Distributed Data Parallel (DDP) 是 PyTorch 中更高效的分布式训练方式，适用于跨多个节点（服务器）的训练。它比 DataParallel 更加高效，因为每个 GPU 拥有一个独立的进程，并且梯度计算和同步效率更高。

### 使用 DistributedDataParallel 进行分布式训练


### DDP 的优势

+ 更高效的梯度同步：每个进程负责自己的一部分数据，减少了不必要的内存开销。

+ 分布式环境支持：支持跨多个机器进行分布式训练。

+ 灵活性：支持多种后端（如 NCCL、Gloo 等）。
