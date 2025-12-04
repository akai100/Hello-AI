torch.compile 是加速你的 PyTorch 代码的新方法！torch.compile 通过将 PyTorch 代码即时编译为优化的内核，在只需最少代码改动的情况下，让 PyTorch 代码运行得更快。

    通过 “即时编译（JIT）” 技术，将零散的 PyTorch 操作转换为经过优化的底层计算内核（减少冗余开销、提升硬件利用率）。
    无需大幅修改原有代码，仅需少量改动就能让代码运行提速，兼顾易用性与性能提升。

torch.compile通过追踪你的Python代码来实现这一点，寻找PyTorch操作。难以追踪的代码会导致图中断，这意味着优化机会的丢失，而非错误或无提示的不正确性。

    torch.compile 的核心逻辑是追踪 Python 代码执行流程，筛选出其中的 PyTorch 操作（如torch.sin、矩阵运算等），
    再将这些操作编译为优化内核以提升速度。图中断不会导致程序报错，也不会出现结果不正确的情况，仅会损失该部分代码的优化可能
    （即这部分代码仍按原生 Python 逻辑执行，无法享受编译加速）

# 1. 基本用法

开启了一些日志记录，以帮助我们了解torch.compile在后台的工作情况。以下代码将打印出torch.compile跟踪的PyTorch操作：

```python3
import torch

torch._logging.set_logs(graph_code=True)
```

```torch.compile``` 是一个装饰器，它可以接收任意的 Python 函数。

```python3
def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(3, 3), torch.randn(3, 3))

@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
```

```torch.compile``` 会递归应用，因此顶级已编译函数中的嵌套函数调用也会被编译。

```python3
def inner(x):
    return torch.sin(x)


@torch.compile
def outer(x, y):
    a = inner(x)
    b = torch.cos(y)
    return a + b


print(outer(torch.randn(3, 3), torch.randn(3, 3)))
```

我们还可以通过调用```torch.nn.Module```实例的```.compile()```方法，或者直接对该模块进行```torch.compile```处理来对其进行优化。这相当于对模块的```__call__```方法（该方法间接调用```forward```）进行```torch.compile```处理。

```python3
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))


mod1 = MyModule()
mod1.compile()
print(mod1(torch.randn(3, 3)))

mod2 = MyModule()
mod2 = torch.compile(mod2)
print(mod2(torch.randn(3, 3)))
```

# 2. 展示加速效果

# 3. 相比 TorchScript 的优势

# 4. 图中断

图中断是```torch.compile```中最基本的概念之一。它允许```torch.compile```通过中断编译、运行不支持的代码，然后恢复编译来处理任意的Python代码。
“图中断”这一术语源于```torch.compile```试图捕获并优化PyTorch操作图这一事实。当遇到不支持的Python代码时，这个图就必须被“中断”。图中断会导致优化机会的丧失，
这可能仍然是不理想的，但总比出现无声的错误或硬崩溃要好。

# 5. 故障排除



