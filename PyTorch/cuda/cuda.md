
torch.cuda用于设置和运行CUDA操作。它会跟踪当前选定的GPU，您分配的所有CUDA张量默认情况下都会在该设备上创建。可以使用```torch.cuda.device```上下文管理器更改选定的设备。

然而，一旦张量被分配，你就可以在其上执行操作，而不受所选设备的影响，并且结果将始终与该张量位于同一设备上。

默认情况下，跨GPU操作是不被允许的，但```copy_()```以及其他具有类似复制功能的方法（如```to()```和```cuda()```）除外。除非启用对等内存访问，否则任何尝试在分布于不同设备的张量上启动操作都会引发错误。


```python3
cuda = torch.device('cuda')     # Default CUDA device
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

with torch.cuda.device(1):
    a = torch.tensor([1., 2.], device=cuda)
    b = torch.tensor([1., 2.]).cuda()
```

##  Ampere(及后续) 设备上的TensorFloat-32（TF32）

在PyTorch 2.9之后，提供了一套新的API，以更精细的方式控制TF32的行为，并建议使用这些新API以获得更好的控制效果。我们可以为每个后端和每个运算符设置float32精度。我们还可以为特定运算符覆盖全局设置。

## FP16 GEMM中的低精度归约

## BF16 GEMM中的低精度归约

## FP16 GEMM中的全FP16累积

## 异步执行

###  CUDA 流

### 反向传播的流语义

## 内存管理器




