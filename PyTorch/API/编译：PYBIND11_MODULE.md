```PYBIND11_MODULE``` 是 Pybind11 库提供的宏定义，核心作用是：

+ 将 C++/CUDA 编写的函数 / 类「绑定」到 Python 环境中，让 Python 可以像调用普通函数一样调用底层算子；

+ 自动处理 Python 和 C++ 之间的类型转换（如 torch.Tensor ↔ at::Tensor、int ↔ int64_t）；

+ 无缝集成到 PyTorch 自定义算子中，是 CUDAExtension 编译后能被 Python 导入的关键；

**1. 基本用法**

```c++
// 绑定到 Python 的核心宏：PYBIND11_MODULE(模块名, 绑定对象)
// 模块名必须和 CUDAExtension 中定义的 name 一致（如 custom_add_ops）
PYBIND11_MODULE(custom_add_ops, m) {
    // 文档字符串（可选，Python中用 help() 可查看）
    m.doc() = "Pybind11 绑定自定义加法算子";

    // 绑定 add 函数：def(Python中函数名, C++函数指针, 文档字符串)
    m.def("add", &add, "一个简单的加法函数",
          // 指定参数名（Python调用时可按名传参）
          pybind11::arg("a"), pybind11::arg("b"));
}
```

**2. 指定参数名 & 绑定类**

```c++
// 格式：PYBIND11_MODULE(模块名, 绑定对象)
// 模块名：必须和 setup.py 中 CUDAExtension 的 name 完全一致（如 custom_add_ops）
// 绑定对象：通常用 m 表示，是 pybind11::module 类型的对象
PYBIND11_MODULE(custom_add_ops, m) {
    // 1. 设置模块文档字符串
    m.doc() = "自定义 CUDA 算子模块";

    // 2. 绑定函数：m.def(参数1, 参数2, 参数3, 参数4...)
    // 参数1：Python 中调用的函数名（如 "vec_add"）
    // 参数2：C++ 函数指针（如 &vec_add_cuda）
    // 参数3：函数文档字符串（可选）
    // 参数4+：指定参数名（可选，方便 Python 按名传参）
    m.def("vec_add", &vec_add_cuda, 
          "CUDA 向量加法",
          pybind11::arg("a"), pybind11::arg("b"));

    // 3. 绑定类（进阶，如自定义算子类）
    // pybind11::class_<MyClass>(m, "MyClass")
    //     .def(pybind11::init<>())  // 绑定构造函数
    //     .def("forward", &MyClass::forward);  // 绑定类方法
}

```

**3. 函数参数默认值**

```c++
// Pybind11 绑定默认参数
m.def("gelu", &gelu_cuda,
      "CUDA GELU 算子",
      pybind11::arg("x"),
      pybind11::arg("approximate") = true);  // 默认参数
```
