
```torch.compile``` 是 **PyTorch 2.0** 引入的**统一模型编译入口**，目标是：

**在尽量不改代码的前提下，让 PyTorch 模型运行得更快**

它通过：

+ 捕获 Python 级别的计算图

+ 对计算图进行优化和融合

+ 调用高性能后端（如 Triton、CUDA、CPU kernel）

实现 自动加速训练和推理。


## 整体架构

```
Python Model
   ↓
TorchDynamo（图捕获）
   ↓
AOTAutograd（前向/反向图拆分）
   ↓
Inductor（算子融合 & Kernel 生成）
   ↓
Backend（CUDA / CPU / Triton）

```

## 核心组件

### TorchDynamo - 图捕获器

作用：

+ 在 Python 层 **拦截字节码**

+ 将 Eager 执行的代码转换为 **FX Graph**

```python3
def f(x, y):
    return x * y + 2

compiled_f = torch.compile(f)
```

TorchDynamo 会：

+ 观察 Python 控制流（if/for）

+ 只要控制流与 Tensor shape / dtype 无关 -> 可编译

+ 否则 -> graph break

graph break 是性能杀手

### FX Graph（中间表示）

TorchDynamo 输出的是 torch.fx.GraphModule

```python3
graph(x, y):
    mul = x * y
    add = mul + 2
    return add
```

特点：

+ Python-level IR

+ 易分析、易修改

+ 为后端优化提供统一接口

### AOTAutograd - 自动微分提前化

AOT = Ahead Of Time

作用：

+ 将训练图拆分为：

  + forward graph
 
  + backward graph
 
+ 提前生成反向传播代码

+ 避免 runtime autograd 开销

### TorchInductor - 核心性能引擎

Inductor 是真正“加速”的地方

功能：

+ 算子融合

+ 内存规划

+ 生成

  + Triton kernel（GPU）
 
  + C++/ OpenMP（CPU）

## ```torch.compile```常用参数

```python3
torch.compile(
    model,
    backend="inductor",
    mode="default",
    fullgraph=False,
    dynamic=False
)
```

+ backend

  + inductor

    默认，性能最好

  + eager

    仅验证，无加速

  + aot_eager

    测试 AOT

+ mode

  + default

    平衡编译时间和性能

  + reduce-overhead

    小模型/小 batch

  + max-autotune

    极致性能（编译慢）

+ fullgraph

  + 强制整个 forward 必须是一个图
  
  + 有 graph break 就报错

  + 适合调试 / 研究

+ dynamic

  + 支持动态 shape
  
  + 编译更复杂

  + 性能略低但更通用

## 调试与分析工具

**1. 打印 graph break**

```
TORCH_LOGS="graph_breaks" python train.py
```

**2. 查看生成代码**

```
TORCH_LOGS="output_code" python train.py
```

**3. 禁用 compile（对比）**

```
torch._dynamo.disable()
```

## 常见不容易被编译代码

### 常见 graph break

```python3
if x.sum() > 0:   # Tensor 参与 Python 控制流
    ...
```

```python3
print(x)         # Python side-effect
```

```python3
x.item()         # Tensor → Python scalar
```

```python3
list.append(x)   # Python 数据结构
```

### 推荐写法

```python3
torch.where(cond, a, b)
```

```python3
torch.nn.functional.*
```

```python3
Tensor 运算替代 Python 逻辑
```

## 什么时候不该用 ```torch.compile```

+ 模型极小（编译时间 > 运行时间）

+ Python 逻辑非常复杂

+ 强依赖动态 shape / Python side-effect

+ 快速原型验证
