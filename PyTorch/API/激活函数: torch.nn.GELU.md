
```python3

torch.nn.GELU(str)

````

+ str： 计算方式

  + "tanh"(默认)

    tanh 近似，用 tanh 拟合误差函数 erf，速度快，精度高

  + "none"

    精确计算，直接调用 erf 函数，稍慢，理论无损失
