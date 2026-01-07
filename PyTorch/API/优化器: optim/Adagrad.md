```python3
torch.optim.Adagrad(params: ParamsT,
                    lr: Union[float, Tensor] = 1e-2,
                    lr_decay: float = 0,
                    weight_decay: float = 0,
                    initial_accumulator_value: float = 0,
                    eps: float = 1e-10,
                    foreach: Optional[bool] = None,
                    maximize: bool = False,
                    differentiable: bool = False,
                    fused: Optional[bool] = None)
```

+ lr

  初始学习率，默认0.01

+ lr_decay

  学习率衰减系数，默认0.0（无衰减）

+ weight_decay

  权重衰减（L2正则化），默认0.0

+ initial_accumulator_value

  梯度平方和的初始值，默认0.0

+ eps

  数值稳定项，避免分母为0，默认1e-10

+ foreach

  是否使用foreach实现，默认None

+ maximize

  是否最大化目标函数，默认False

+ differentiable

  是否支持求导（仅Autograd），默认False

## Adagrad 详解

1. 梯度平方和累积

 $G_t[i]=G_{t-1}[i] + (g_t[i])^2 + initial_accumulator_value$

其中：$G_t[i]：第 $t$轮参数 $i$的梯度平方和, $g_t[i]$：第 $t$ 轮参数 $i$ 的梯度

2. 学习率衰减

 $\eta_t = \frac{\text{lr}}{1 + t \cdot \text{lr\\_decay}}$

3. 参数更新（含权重衰减）

$\theta_{t+1}[i]=\theta_t[i]-\eta_t \cdot \frac{g_t[i]+\lambda \cdot \theta_t[i]}{sqrt{G_t[i]}+\eps}$
