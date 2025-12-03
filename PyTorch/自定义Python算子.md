PyTorch 提供了大量可对张量进行操作的算子（例如 ```torch.add```、```torch.sum`` 等）。不过，你可能希望在 PyTorch 中使用一种新的自定义算子，
这种算子或许是由第三方库编写的。本教程将展示如何对 Python 函数进行封装，使其能像 PyTorch 原生算子一样工作。在 PyTorch 中创建自定义算子的原因包括：

+ 对于```torch.compile```，将任意Python函数视为不透明的可调用对象（也就是说，阻止```torch.compile```追踪该函数内部）。

+ 向任意Python函数添加训练支持

使用```torch.library.custom_op()```创建Python自定义运算符。使用C++TORCH_LIBRARYAPI创建C++自定义运算符（这些工作在无Python环境中）。

# 1. 示例：将PIL的crop包装到自定义运算符中

# 2. 为crop添加训练支持
