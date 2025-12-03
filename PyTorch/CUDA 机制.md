
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

每个流内部的操作会按照其创建顺序进行序列化，但不同流的操作可以以任何相对顺序并发执行，除非使用了显式的同步函数（例如synchronize()或wait_stream()）。

```python3
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
with torch.cuda.stream(s):
    # sum() may start execution before normal_() finishes!
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
    loss.backward()
    use grads

# Unsafe
with torch.cuda.stream(s):
    loss.backward()
use grads

# Safe, with synchronization
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)
use grads

# Safe, populating initial grad and invoking backward are in the same stream context
with torch.cuda.stream(s):
    loss.backward(gradient=torch.ones_like(loss))

# Unsafe, populating initial_grad and invoking backward are in different stream contexts,
# without synchronization
initial_grad = torch.ones_like(loss)
with torch.cuda.stream(s):
    loss.backward(gradient=initial_grad)

# Safe, with synchronization
initial_grad = torch.ones_like(loss)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    initial_grad.record_stream(s)
    loss.backward(gradient=initial_grad)
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
