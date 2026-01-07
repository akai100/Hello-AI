```python3
torch.optim.RMSprop(params: ParamsT,
                    lr: Union[float, Tensor] = 1e-2,
                    alpha: float = 0.99
                    eps: float = 1e-8,
                    weight_decay: float = 0,
                    momentum: float = 0,
                    centered: bool = False,
                    capturable: bool = False,
                    foreach: Optional[bool] = None,
                    maximize: bool = False,
                    differentiable: bool = False)
```

+ params

  待优化的参数组

+ lr

  初始学习率，默认0.001

+ alpha

  梯度平方的指数衰减系数，默认0.99

+ eps

  数值稳定项，避免分母为0，默认1e-8

+ weight_decay

  权重衰减（L2正则化），默认0.0

+ momentum

  动量系数，默认0.0（无动量）

+ centered

  是否使用中心化梯度，默认False

+ foreach

  是否使用foreach实现，默认None

+ maximize

  是否最大化目标函数，默认False

+ differentiable

  是否支持求导（仅Autograd），默认False

## 参数更新公式

**1. 梯度平方的指数移动平均**

$E[g^2]_t=\alpha \cdot E[g^2]_{t-1}+(1-\alpha) \cdot f_{t}^{2}$

**2. 中心化修正（centered=True 时有效）**

 $E[g]_t=\alpha \cdot E[g]_{t-1}+(1-\alpha) \cdot g_t$

 $std_t = \sqrt{}$

**3. 无中心化时的均方根（默认）**

 $std_t=\sqrt{E[g^2]_t+\eps}$

**4. 动量更新**

 $v_t=momentum \cdot v_{t-1} + \frac{\eta}{std_t} \cdot (g_t + \gamma \cdot \theta_t)$

**5. 参数最终更新（含 weight_decay，λ 为权重衰减系数）**

 $\theta_{t+1}=\theta-v_t$



