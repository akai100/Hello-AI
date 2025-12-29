```torch.version``` 是 PyTorch 顶层辅助模块，核心作用是**查询 PyTorch 框架本身及相关依赖库（如 CUDA、cuDNN 等）的版本信息**，
为版本兼容性排查、环境验证、部署适配提供关键参考依据，是 PyTorch 开发与运维中的常用工具模块。


+ torch.version.__version__

  PyTorch 框架自身的核心版本号（最常用属性）

  格式通常为 x.y.z（如 2.4.0），部分版本附带后缀（如 2.4.0+cu121 标识适配 CUDA 12.1）

+ torch.version.cuda

  PyTorch 编译时所使用的 CUDA 工具包版本号

  仅 GPU 版 PyTorch 有有效值，CPU 版返回 None 或空字符串

+ torch.version.cudnn

  PyTorch 集成的 cuDNN 库版本号（cuDNN 是 NVIDIA GPU 加速深度学习的核心库）

  仅 GPU 版 PyTorch 且配置 cuDNN 后有有效值，CPU 版返回 None

+ torch.version.mps

  PyTorch 对 Apple M 系列芯片（MPS）的支持版本

  仅 macOS 系统且支持 MPS 的 PyTorch 有有效值

+ torch.version.git_version

  PyTorch 编译对应的 Git 提交哈希值

  用于排查特定编译版本的源码差异，一般仅开发 / 调试场景使用
