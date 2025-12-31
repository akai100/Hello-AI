
在 PyTorch 中，torch.profiler 是一个强大的工具，用于性能分析。它可以帮助开发者分析模型的训练过程，识别性能瓶颈，优化模型的计算效率。
torch.profiler 提供了详细的时间、内存、CUDA 事件、模型层级等信息，以便用户了解模型运行时的行为。

## 1. 概述

torch.profiler 允许你记录和分析训练过程中各个操作的性能表现。通过对训练过程中的每个步骤进行时间追踪，开发者可以识别出影响训练速度的瓶颈，
并据此优化模型。Profiler 还可以帮助在多线程、多GPU 环境下进行性能分析。

torch.profiler 提供了不同的方式来采集性能数据，包括操作的执行时间、内存使用情况、操作的总数等。

## 2. 基本用法

### 2.1 ```torch.profiler.profile```

```torch.profiler.profile``` 是性能分析的核心类，允许用户在模型训练期间进行性能记录。这个类可以收集并返回模型运行时的各种信息，包括时间、内存、CUDA 时间等。

```python

```

### 2.2 ```torch.profiler.profile``` 的关键参数



### 2.3 基本示例

```python3
import torch
import torch.profiler

# 模拟一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleModel()

# 输入张量
input_tensor = torch.randn(64, 10)

# 开始性能分析
with torch.profiler.profile(activities=[
    torch.profiler.ProfilerActivity.CPU, 
    torch.profiler.ProfilerActivity.CUDA
]) as prof:
    model(input_tensor)

# 打印性能报告
prof.export_chrome_trace("trace.json")

```

### 2.4 性能数据可视化

将追踪的性能数据导出为 Chrome trace 格式后，可以使用 Chrome 浏览器的 about:tracing
 功能，加载 trace.json 文件并进行可视化。

 ## 3. 高级功能

 ### 3.1 ```torch.profiler.schedule```

torch.profiler.schedule 允许你指定何时开始和结束性能分析，它支持设置具体的步数、时间间隔或其他条件来控制性能分析的执行。

### 3.2 ```torch.profiler.Profiler``` 输出

```torch.profiler.Profiler``` 记录的输出包括：

+ self_cpu_time_total: 每个操作的 CPU 总时间。

+ cpu_memory_usage: 操作在 CPU 上的内存使用情况。

+ cuda_time_total: 每个操作的 CUDA 总时间。

+ cuda_memory_usage: 每个操作的 CUDA 内存使用情况。

+ name: 操作的名称（如矩阵乘法、卷积等）。

+ stack: 调用堆栈，用于显示操作来源。

可以使用 prof.key_averages() 方法获取总结的统计信息，按操作进行排序并分析瓶颈：

```python3
# 获取分析结果并按时间排序
prof.key_averages().table(sort_by="cpu_time_total")
```

### 3.3 torch.profiler.tensorboard

torch.profiler.tensorboard 支持将性能数据导出为 TensorBoard 格式，方便通过 TensorBoard 可视化分析。

```python3
from torch.profiler.tensorboard import SummaryWriter

with torch.profiler.profile(activities=[
    torch.profiler.ProfilerActivity.CPU, 
    torch.profiler.ProfilerActivity.CUDA
]) as prof:
    model(input_tensor)

# 导出到 TensorBoard
prof.export_chrome_trace("trace.json")
writer = SummaryWriter()
writer.add_graph(model, input_tensor)
writer.close()

```
 
