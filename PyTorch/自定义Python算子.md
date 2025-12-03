PyTorch 提供了大量可对张量进行操作的算子（例如 ```torch.add```、```torch.sum`` 等）。不过，你可能希望在 PyTorch 中使用一种新的自定义算子，
这种算子或许是由第三方库编写的。本教程将展示如何对 Python 函数进行封装，使其能像 PyTorch 原生算子一样工作。在 PyTorch 中创建自定义算子的原因包括：

+ 对于```torch.compile```，将任意Python函数视为不透明的可调用对象（也就是说，阻止```torch.compile```追踪该函数内部）。

+ 向任意Python函数添加训练支持

使用```torch.library.custom_op()```创建Python自定义运算符。使用C++TORCH_LIBRARYAPI创建C++自定义运算符（这些工作在无Python环境中）。

# 1. 示例：将PIL的crop包装到自定义运算符中

```python3
import torch
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import PIL
import IPython
import matplotlib.pyplot as plt

def crop(pic, box):
    img = to_pil_image(pic.cpu())
    cropped_img = img.crop(box)
    return pil_to_tensor(cropped_img).to(pic.device) / 255.

def display(img):
    plt.imshow(img.numpy().transpose((1, 2, 0)))
```

```crop```无法通过```torch.compile```开箱即用地得到有效处理：```torch.compile```会在遇到无法处理的函数时引发“图中断”，而图中断会对性能产生不利影响。
以下代码通过触发错误展示了这一点（当发生图中断时，启用```fullgraph=True```的```torch.compile```会引发错误）。

```python3
@torch.compile(fullgraph=True)    # PyTorch 的编译装饰器，用于将 Python 函数编译为优化后的计算图以提升执行效率。fullgraph=True 是严格模式，要求函数执行过程中
def f(img):
    return crop(img, (10, 10, 50, 50))

# 未封装为 PyTorch 自定义算子的 crop 函数，内部包含 PyTorch 无法追踪的操作（示例中是 PIL 库的图像裁剪逻辑）：
# cropped_img = f(img)

```

为了将crop作为黑盒用于torch.compile，我们需要做两件事:

+ 将该函数包装成PyTorch自定义算子;

+ 向算子添加一个“FakeTensor内核”（又名“元内核”）。给定一些FakeTensors输入（没有存储的虚拟张量），此函数应返回具有正确张量元数据（形状/步长/dtype/设备）的所选虚拟张量;

```python3
from typing import Sequence

# Use torch.library.custom_op to define a new custom operator.
# If your operator mutates any input Tensors, their names must be specified
# in the ``mutates_args`` argument.
@torch.library.custom_op("mylib::crop", mutates_args=())           # 把普通 Python 函数crop注册为 PyTorch 自定义算子，命名空间为mylib::crop（避免命名冲突）
def crop(pic: torch.Tensor, box: Sequence[int]) -> torch.Tensor:
    img = to_pil_image(pic.cpu())
    cropped_img = img.crop(box)
    return (pil_to_tensor(cropped_img) / 255.).to(pic.device, pic.dtype)

# Use register_fake to add a ``FakeTensor`` kernel for the operator
@crop.register_fake                # 为自定义算子添加 “虚拟张量内核”，供torch.compile追踪时使用 ——torch.compile需要提前知晓算子输入输出的张量元数据（形状、设备、 dtype 等），无需实际执行计算（避免图中断）。
def _(pic, box):
    channels = pic.shape[0]
    x0, y0, x1, y1 = box
    result = pic.new_empty(y1 - y0, x1 - x0, channels).permute(2, 0, 1)
    # The result should have the same metadata (shape/strides/``dtype``/device)
    # as running the ``crop`` function above.
    return result
```

# 2. 为crop添加训练支持

使用```torch.library.register_autograd```为算子添加训练支持。相比直接使用```torch.autograd.Function```，更推荐这种方式；
```autograd.Function```与PyTorch算子注册API的某些组合在与```torch.compile```结合使用时，可能会（并且已经导致）出现无提示的错误。

        使用 torch.library.register_autograd 为 PyTorch 自定义算子添加训练支持，核心是为算子定义反向传播逻辑，使其能兼容 PyTorch 的自动求导（autograd）系统，关键要点如下：
        核心目的：让自定义算子支持梯度计算，满足训练场景需求（替代直接使用 torch.autograd.Function，避免与 torch.compile 结合时出现隐性错误）。
        1. 核心步骤：
        定义 backward 函数：接收上游梯度（grad_output），计算并返回输入张量的梯度（grad_input），逻辑需基于 PyTorch 可识别的算子（或已封装的自定义算子）。
        定义 setup_context 函数：保存正向传播时的关键信息（如输入形状、参数等），供反向传播时使用。
        通过 算子.register_autograd(backward, setup_context=setup_context) 完成注册。
        关键约束：
        反向传播逻辑（backward 函数）必须由 PyTorch 可识别的算子组成（不能直接使用第三方无梯度支持的函数，需先封装为自定义算子）。
        若算子无需训练支持，可省略该注册；未注册却用于训练时，PyTorch 会抛出错误

如果你不需要训练支持，就无需使用```torch.library.register_autograd```。如果你最终使用没有自动求导注册的custom_op进行训练，我们会抛出一条错误信息。

crop的梯度公式本质上是```PIL.paste``（我们将把推导过程留给读者作为练习）。让我们首先将```paste``封装到一个自定义算子中

```python3
@torch.library.custom_op("mylib::paste", mutates_args=())
def paste(im1: torch.Tensor, im2: torch.Tensor, coord: Sequence[int]) -> torch.Tensor:
    assert im1.device == im2.device
    assert im1.dtype == im2.dtype
    im1_pil = to_pil_image(im1.cpu())
    im2_pil = to_pil_image(im2.cpu())
    PIL.Image.Image.paste(im1_pil, im2_pil, coord)
    return (pil_to_tensor(im1_pil) / 255.).to(im1.device, im1.dtype)

@paste.register_fake
def _(im1, im2, coord):
    assert im1.device == im2.device
    assert im1.dtype == im2.dtype
    return torch.empty_like(im1)
```

现在让我们使用register_autograd来指定crop的梯度公式：

```python3
def backward(ctx, grad_output):
    grad_input = grad_output.new_zeros(ctx.pic_shape)
    grad_input = paste(grad_input, grad_output, ctx.coords)
    return grad_input, None

def setup_context(ctx, inputs, output):
    pic, box = inputs
    ctx.coords = box[:2]
    ctx.pic_shape = pic.shape

crop.register_autograd(backward, setup_context=setup_context)
```



