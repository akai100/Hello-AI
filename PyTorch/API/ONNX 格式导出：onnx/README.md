
onnx 是一个开放的神经网络交换格式（Open Neural Network Exchange），用于在不同的深度学习框架之间共享模型。onnx 允许你将 PyTorch 模型导出为 onnx 格式，
并在其他支持 ONNX 的平台和框架（如 TensorFlow、Caffe2、MXNet 等）上进行推理。

在 PyTorch 中，torch.onnx 模块提供了导出和加载 ONNX 模型的功能。你可以使用这个模块将训练好的 PyTorch 模型转换为 ONNX 格式，从而实现跨平台部署和推理。


## 1. 导出 PyTorch 模型为 ONNX 格式

```torch.onnx.export()``` 函数是导出 PyTorch 模型为 ONNX 格式的核心方法。它将 PyTorch 模型的计算图（包括权重）保存为 ONNX 格式文件，供其他框架（如 TensorFlow、Caffe2、MXNet 等）使用。

### 1.1 torch.onnx.export() 函数

## 2. 导入 ONNX 模型

PyTorch 本身不直接提供将 ONNX 模型加载到 PyTorch 中的功能。但您可以使用 ONNX Runtime 进行推理，也可以在一些特定场景下通过 ONNX 模型继续训练。
在 PyTorch 中加载和推理 ONNX 模型需要借助 onnx 和 onnxruntime 库。


## 3. ONNX 模型优化

PyTorch 支持通过 ```torch.onnx``` 导出模型时进行一些优化，例如常量折叠和图优化。

### 3.1 常量折叠

常量折叠是通过将计算图中的常量部分（例如矩阵乘法中的静态权重）折叠成结果，从而优化计算图。这通常会提高推理速度。

### 3.2 其他优化

+ 去除冗余节点

  ONNX 在导出时会自动去除计算图中的冗余节点，减少图的复杂度

+ 图形简化

  使用 onnx-simplifier 等工具可以进一步简化 ONNX 模型中的操作，减少计算量


## 4. 动态轴

dynamic_axes 参数用于设置输入和输出张量的哪些维度是动态的。例如，batch_size 经常是动态的，因此可以将其设置为动态轴。

### 4.1 设置动态轴

