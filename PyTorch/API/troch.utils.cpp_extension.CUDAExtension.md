```CUDAExtension``` 是 PyTorch 封装的 CUDA/C++ 混合编译工具类，继承自 Python ```setuptools.Extension```，核心作用是：

+ 自动处理 CUDA 核函数 (.cu) 和 C++ 封装层（.cpp）的编译流程；

+ 适配不同平台（Linux/Windows）、不同 CUDA 版本的编译规则；

+ 生成可被 PyTorch 直接调用的 Python 扩展模块（.so/.pyd 文件）

```
CUDAExtension(name, sources, extra_compile_args, include_dirs, library_dirs, libraries, define_macros, undef_macros)
```

+ name

  str，编译后的扩展模块名

+ sources

  list[str]，指定待编译的源文件（.cpp/.cu）

+ extra_compile_args

  dict，传给编译器的额外参数（分 nvcc/cxx）

+ include_dirs

  指定头文件路径（解决找不到 CUDA/PyTorch 头文件）

+ library_dirs

  list[str]，指定库文件名称（如CUDA/cuDNN库）

+ libraries

  list[str]，指定要链接的库（如 cudart/cublas/cudnn）

+ define_macros

  list[tuple]，定义编译宏（控制条件编译）

+ undef_macros

  list[str]，取消编译宏
