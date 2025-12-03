
torch.cuda用于设置和运行CUDA操作。它会跟踪当前选中的GPU，你分配的所有CUDA张量默认都会在该设备上创建。可以使用torch.cuda.device上下文管理器更改选中的设备。

然而，一旦张量被分配，你就可以在其上执行操作，而无需考虑所选设备，并且结果将始终与该张量位于同一设备上。

默认情况下不允许跨GPU操作，但copy_()以及其他具有类似复制功能的方法（如to()和cuda()）除外。除非启用对等内存访问，否则任何尝试在分布于不同设备上的张量上运行操作都会引发错误。

# 1. Ampere（及后续）设备上的TensorFloat-32（TF32）

在PyTorch 2.9之后，我们提供了一套新的API，以更精细的方式控制TF32的行为，并建议使用这些新API以获得更好的控制效果。我们可以为每个后端和每个算子设置float32精度。我们还可以为特定算子覆盖全局设置。

    TF32 是一种 “折中精度”（NVIDIA 推出的混合精度格式），兼顾 float32 的精度和 float16 的速度，常用于 GPU 上的
    矩阵乘法（matmul）、卷积（conv）等计算密集型算子，旧版 PyTorch 对 TF32 的控制多是 “全局开关”（比如全局启用 / 禁用），不够灵活。
    2. 新 API 的核心改进点
    （1）控制粒度更细：按 “后端 + 算子” 拆分
    按后端：支持为不同计算后端（比如 CUDA、CPU，核心是 GPU 后端）单独设置 TF32 行为（比如仅让 CUDA 后端的算子使用 TF32，CPU 保持纯 float32）；
    按算子：不再是 “所有算子统一规则”，而是可以针对单个算子（比如 torch.matmul、torch.nn.Conv2d）单独配置 float32/TF32 精度（
    比如让卷积用 TF32 提速，矩阵乘法保持纯 float32 保精度）。
    （2）支持 “全局默认 + 局部覆盖”
    先设置全局规则（比如默认所有 CUDA 算子启用 TF32）；
    对需要特殊处理的算子，单独设置局部规则覆盖全局（比如全局启用 TF32，但某个关键的 matmul 算子必须用纯 float32，直接为该算子配置即可）。
    3. 核心价值
    平衡 “速度” 和 “精度”：无需为了全局提速牺牲关键算子的精度，也无需为了保精度放弃非关键算子的性能收益；
    适配复杂场景：比如混合精度训练中，部分算子对精度敏感（如损失计算、梯度聚合），部分算子（如特征提取的卷积）可容忍 TF32，新 API 
    能精准匹配这种需求。

```python3
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"
```

从PyTorch 1.7版本开始，出现了一个名为allow_tf32的新标志。在PyTorch 1.7至PyTorch 1.11版本中，该标志默认值为True，而在PyTorch 1.12及更高版本中，默认值为False。
此标志用于控制PyTorch是否被允许在内部使用TensorFloat32（TF32）张量核心（自Ampere架构起在NVIDIA GPU上可用）来计算矩阵乘法（矩阵相乘和批处理矩阵相乘）和卷积运算。

TF32张量核心旨在通过将输入数据舍入为具有10位尾数，并以FP32精度累积结果、保持FP32动态范围，从而在torch.float32张量的矩阵乘法和卷积运算上实现更好的性能。

矩阵乘法和卷积是分开控制的，它们相应的标志可在以下位置访问：
```python3
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
```

矩阵乘法的精度也可以通过set_float32_matmul_precision()进行更广泛的设置（不仅限于CUDA）。请注意，除了矩阵乘法和卷积本身外，内部使用矩阵乘法或卷积的函数和神经网络模块也会受到影响。
这些包括nn.Linear、nn.Conv*、cdist、tensordot、仿射网格和网格采样、自适应对数softmax、GRU和LSTM。

# 2. FP16 GEMM中的低精度归约

fp16通用矩阵乘法（GEMM）可能会通过一些中间的降低精度的缩减操作来实现（例如，使用fp16而非fp32）。这种有选择性的精度缩减能够在某些工作负载（尤其是那些具有大k维度的工作负载）
和GPU架构上实现更高的性能，但代价是数值精度的降低以及可能出现的溢出问题。

如果需要全精度归约，用户可以通过以下方式在fp16通用矩阵乘法中禁用低精度归约：

```python3
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
```

# 3. BF16 GEMM中的降低精度缩减

对于BFloat16的通用矩阵乘法，也存在一个类似（如上所述）的标志。请注意，对于BF16，此开关默认设置为True，如果你在工作负载中观察到数值不稳定性，可能希望将其设置为False。

如果不希望使用降低精度的归约操作，用户可以通过以下方式在bf16通用矩阵乘法（GEMMs）中禁用降低精度的归约操作：

```python3
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```


# 4. FP16 GEMM中的全FP16累加

某些GPU在以FP16进行所有FP16 GEMM累加时性能会有所提升，但这会以牺牲数值精度和增加溢出可能性为代价。请注意，此设置仅对计算能力为7.0（伏特架构）或更新的GPU有效。

可通过以下方式启用此行为：

```python3
torch.backends.cuda.matmul.allow_fp16_accumulation = True
```

# 5. 异步执行

默认情况下，GPU 操作是异步的。当你调用一个使用 GPU 的函数时，这些操作会被排入特定设备的队列，但不一定会立即执行。这使我们能够并行执行更多计算，包括在 CPU 或其他 GPU 上的操作。

一般来说，异步计算的效果对调用者是不可见的，因为（1）每个设备都按照操作排队的顺序执行操作，以及（2）PyTorch在CPU与GPU之间或两个GPU之间复制数据时会自动执行必要的同步。因此，计算的进行方式就如同每个操作都是同步执行的一样。
        

你可以通过设置环境变量CUDA_LAUNCH_BLOCKING=1来强制同步计算。当GPU上发生错误时，这可能会很有用（在异步执行的情况下，此类错误要到操作实际执行后才会报告，因此堆栈跟踪不会显示请求该操作的位置）。

异步计算的一个结果是，没有同步的时间测量是不准确的。要获得精确的测量结果，应该在测量前调用```torch.cuda.synchronize()```，或者使用```torch.cuda.Event```来记录时间，如下所示：

```python3

```

## 5.1 CUDA 流

一个CUDA流是属于特定设备的线性执行序列。通常情况下，你无需显式创建它：默认情况下，每个设备都使用自己的“默认”流。

        1. 核心属性：设备专属的线性执行序列
        「属于特定设备」：CUDA 流不能跨 GPU 设备使用，比如 GPU 0 的流只能调度该设备上的核函数、内存拷贝等操作，无法操作 GPU 1 的资源，这是由 CUDA 的设备隔离模型决定的。
        「线性执行序列」：同一流内的操作严格按顺序执行（前一个操作完成，后一个才开始），避免了同一流内的并发冲突；而不同流之间的操作可并行（若设备支持）
        每个流内部的操作会按照其创建顺序进行序列化，但不同流的操作可以以任何相对顺序并发执行，除非使用了显式的同步函数（例如synchronize()或wait_stream()）。

        2. 每个 GPU 设备都会默认自带一个「默认流」（也叫 NULL 流），写代码时如果不特意指定自定义流，所有 GPU 相关操作都会自动放进这个默认流里执行，降低入门门槛。

每个流内部的操作会按照其创建顺序进行序列化，但不同流的操作可以以任何相对顺序并发执行，除非使用了显式的同步函数（例如synchronize()或wait_stream()）。例如，以下代码是不正确的：

```python3
cuda = torch.device('cuda')
s = torch.cuda.Stream()                                       # 创建一个新的 CUDA 流
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)    # 在GPU上创建空张量，并用正态分布初始化，normal_ 是in-place操作，在GPU上异步执行
with torch.cuda.stream(s):                                    # j进入指定流下文，后续 CUDA 操作会提交到流 s执行
    # sum(A) 可能在 nomal 完成之前开始执行
    B = torch.sum(A)
```

当“当前流”是默认流时，PyTorch会在数据移动时自动执行必要的同步，如上所述。然而，在使用非默认流时，确保适当同步则是用户的责任。此示例的固定版本如下：

```python3
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
s.wait_stream(torch.cuda.default_stream(cuda))  # NEW!
with torch.cuda.stream(s):
    B = torch.sum(A)
A.record_stream(s)  # NEW!
```

有两处新增内容。torch.cuda.Stream.wait_stream()调用确保在我们开始在副流上运行sum(A)之前，normal_()的执行已经完成。torch.Tensor.record_stream()（详见更多细节）
确保在sum(A)完成之前，我们不会释放A。你也可以在之后的某个时间点，通过torch.cuda.default_stream(cuda).wait_stream(s)手动等待该流（注意，立即等待是没有意义的，
因为这会阻止流的执行与默认流上的其他工作并行运行）。

请注意，即使不存在读取依赖，这种同步也是必要的，例如在本示例中所看到的情况：

```python3
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((100, 100), device=cuda)
s.wait_stream(torch.cuda.default_stream(cuda))  # STILL REQUIRED!
with torch.cuda.stream(s):
    A.normal_(0.0, 1.0)
    A.record_stream(s)
```

尽管对```s```的计算不会读取```A```的内容，且```A```没有其他用途，但仍然需要进行同步，因为```A```可能对应于CUDA缓存分配器重新分配的内存，而旧（已释放）内存中存在未完成的操作。

## 5.2 反向传播的流机制

每个反向CUDA操作都在与其对应的正向操作所使用的相同流上运行。如果你的正向传播在不同流上并行运行独立操作，这将有助于反向传播利用相同的并行性。

        1. 核心规则
        反向传播中每个 “反向操作”（如 backward 对应正向 forward），会默认复用其正向操作所用的 同一个 CUDA 流（stream）—— 流是 CUDA 中管理任务并行 / 串行的核心：同一流内任务按顺序执行，不同流间任务可并行。
        2. 关键价值（正向多流并行时的收益）
        如果正向传播（forward）中，你把独立无关的操作分配到了不同流上（比如多个分支、多个样本的特征提取并行执行），那么反向传播（backward）时：
        （1）每个反向操作会自动进入对应正向操作的流；
        （2）原本正向并行的独立操作，其反向操作也会在各自流上自然并行，无需额外手动分配流。

相对于周围操作，反向调用的流语义与任何其他调用的流语义相同。反向传递会插入内部同步，以确保即使反向操作如前段所述在多个流上运行时也是如此。
更具体地说，当调用```autograd.backward```、```autograd.grad```或```tensor.backward```，并可选地提供CUDA张量作为初始梯度
（例如，```autograd.backward(..., grad_tensors=initial_grads)```、```autograd.grad(..., grad_outputs=initial_grads)```或```tensor.backward(..., gradient=initial_grad)```）时，这些行为

1. 可选地填充初始梯度，

2. 调用反向传播，以及

3. 使用梯度

```python3
s = torch.cuda.Stream()

# Safe, grads are used in the same stream context as backward()
with torch.cuda.stream(s):
    loss.backward()    # 1. 在流s中执行反向传播，生成的梯度张量归属于流s
    use grads          # 2. 同一流上下文内，梯度已在s中完成计算，且归属一致，安全

# Unsafe
with torch.cuda.stream(s):
    loss.backward()    # 1. 流s中异步执行反向传播（未阻塞当前默认流）
use grads              # 2. 在默认流中使用梯度：
                       # - 梯度归属流s，但默认流未等待s完成，可能梯度还没计算完；
                       # - 跨流访问未同步的张量，数据竞争风险

# 安全：跨流 + 显式同步
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)    # 1. 让当前默认流“等待流s完成所有操作”
use grads                                     # 2. 同步后，梯度已计算完，跨流访问安全

#  安全：同流内生成 + 使用初始梯度
with torch.cuda.stream(s):
    # 1. 在流s中创建初始梯度张量（归属于s），同时在s中执行backward()
    loss.backward(gradient=torch.ones_like(loss))    # 用户传入的 gradient 张量是在流 s 的上下文内创建的，其归属流就是 s，与 backward() 执行流一致，无需额外同步。

# 不安全：跨流创建初始梯度（无同步）
initial_grad = torch.ones_like(loss)    # 1. 在默认流中创建，归属流是默认流
with torch.cuda.stream(s):
    # 2. 在流s中执行backward()，但传入的initial_grad归属默认流
    loss.backward(gradient=initial_grad)

# 安全：跨流创建初始梯度（加同步 + 记录流）
initial_grad = torch.ones_like(loss)          # 1. 默认流中创建，归属默认流
s.wait_stream(torch.cuda.current_stream())    # 2. 让流s等待默认流：确保initial_grad已创建完成
with torch.cuda.stream(s):
    initial_grad.record_stream(s)             # 3. 给initial_grad“记录流s”：允许s访问它
    loss.backward(gradient=initial_grad)      # 4. 流s中安全使用initial_grad
```

### 5.2.1 BC 注释：在默认流上使用梯度

在PyTorch的早期版本（1.9及更早版本）中，自动求导引擎总是将默认流与所有反向操作同步，因此以下模式：

```python3
with torch.cuda.stream(s):
    loss.backward()
use grads
```
只要```use grads```在默认流上运行，就是安全的。在当前的PyTorch中，这种模式不再安全。如果```backward()```和```use grads``处于不同的流上下文中，则必须同步这些流：

```python3
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)
use grads
```

# 6. 内存管理

PyTorch使用缓存内存分配器来加快内存分配速度。这使得无需设备同步就能快速释放内存。不过，分配器管理的未使用内存在nvidia-smi中仍会显示为已使用状态。你可以使用```memory_allocated()```
和```max_memory_allocated()```来**监控张量占用的内存**，使用```memory_reserved()```和```max_memory_reserved()``来**监控缓存分配器管理的总内存量**。
调用```empty_cache()```会从PyTorch中释放所有未使用的缓存内存，以便其他GPU应用程序可以使用这些内存。但是，张量占用的GPU内存不会被释放，因此这无法增加PyTorch可用的GPU内存量。

## 6.1 使用PYTORCH_CUDA_ALLOC_CONF优化内存使用

使用缓存分配器可能会干扰诸如cuda-memcheck之类的内存检查工具。要使用cuda-memcheck调试内存错误，请在环境中设置```PYTORCH_NO_CUDA_MEMORY_CACHING=1```以禁用缓存。

缓存分配器的行为可以通过环境变量```PYTORCH_CUDA_ALLOC_CONF```来控制。其格式为```PYTORCH_CUDA_ALLOC_CONF=<option>:<value>,<option2>:<value2>...```可用选项

+ ```backend```允许选择底层分配器实现。目前，有效的选项有native（使用PyTorch的原生实现）和cudaMallocAsync（使用CUDA的内置异步分配器）。cudaMallocAsync需要CUDA 11.4或更高版本。
  默认选项是native。backend适用于进程使用的所有设备，且不能按设备单独指定;

+ max_split_size_mb 可防止原生分配器拆分大于此大小（以MB为单位）的块。这可以减少碎片化，并可能使一些临界工作负载在不耗尽内存的情况下完成。根据分配模式的不同，性能成本可能从“零”到“显著”不等。
  默认值为无限制，即所有块都可以被拆分。```memory_stats()``` 和 ```memory_summary()``` 方法对于调优很有用。此选项应作为因“内存不足”而中止且显示大量非活动拆分块的工作负载的最后手段。
  ```max_split_size_mb```仅在 ```backend:native``` 时才有意义。在 ```backend:cudaMallocAsync``` 情况下，```max_split_size_mb``` 会被忽略;

+ ```roundup_power2_divisions``` 有助于将请求的分配大小向上取整到最接近的2的幂次分区，从而更好地利用块。在原生CUDACachingAllocator中，大小会以512的块大小的倍数向上取整，因此这对于较小的大小来说效果很好。
  然而，对于较大的邻近分配，这可能效率不高，因为每个分配都会使用不同大小的块，这些块的重用率会降到最低。这可能会产生大量未使用的块，并浪费GPU内存容量。此选项允许将分配大小向上取整到最接近的2的幂次分区。例如，
  如果我们需要对1200的大小进行向上取整，且分区数为4，那么1200介于1024和2048之间，如果我们在它们之间进行4次分区，得到的值为1024、1280、1536和1792。因此，1200的分配大小会向上取整到1280，因为这是最接近的2的幂次分区上限值。
  可以指定一个单一值应用于所有分配大小，或者指定一组键值对，为每个2的幂次区间单独设置2的幂次分区数。例如，要为所有256MB以下的分配设置1个分区，为256MB到512MB之间的分配设置2个分区，为512MB到1GB之间的分配设置4个分区，
  为任何更大的分配设置8个分区，请将该参数值设置为：[256:1,512:2,1024:4,>:8]。```roundup_power2_divisions``` 仅在 ```backend:native``` 时有效。在 ```backend:cudaMallocAsync``` 时，```roundup_power2_divisions``` 会被忽略。

+ max_non_split_rounding_mb 允许非拆分块以实现更好的重用，例如
  一个1024MB的缓存块可以重新用于512MB的分配请求。在默认情况下，我们只允许非拆分块最多有20MB的取整，因此512MB的块只能由512-532MB大小的块来提供。如果我们将此选项的值设置为1024，就会允许512-1536MB大小的块用于512MB的块，
  这会提高更大块的复用率。这也将有助于减少延迟，避免代价高昂的cudaMalloc调用。

## 6.2 使用CUDA的自定义内存分配器




