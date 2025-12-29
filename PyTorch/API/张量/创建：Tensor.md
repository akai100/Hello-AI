**1. 基础创建方法（直接生成）**

```python3
# 1. 全0/全1/全值张量
torch.zeros(2, 3)          # 2×3全0
torch.ones(2, 3)           # 2×3全1
torch.full((2, 3), 5.0)    # 2×3全5.0

# 2. 随机张量（核心）
torch.rand(2, 3)           # 0~1均匀分布
torch.randn(2, 3)          # 标准正态分布（均值0，方差1）
torch.randint(0, 10, (2, 3)) # 0~9整数随机

# 3. 单位矩阵/对角矩阵
torch.eye(3)               # 3×3单位矩阵
torch.diag(torch.tensor([1,2,3])) # 对角矩阵
```

**2. 从其他数据转换**

```python3
# 1. 从Python列表/元组转换
x = torch.tensor([[1,2], [3,4]])  # 推荐（自动推断类型）
x = torch.Tensor([[1,2], [3,4]])  # 旧写法（默认float32）

# 2. 从NumPy数组转换（共享内存！）
import numpy as np
np_arr = np.array([[1,2], [3,4]])
x = torch.from_numpy(np_arr)      # 转换为Tensor（类型与np一致）
np_arr2 = x.numpy()               # Tensor转回NumPy（共享内存）

# 注意：共享内存意味着修改一方会影响另一方，避免：
x = torch.tensor(np_arr.copy())   # 深拷贝，解除共享
```
