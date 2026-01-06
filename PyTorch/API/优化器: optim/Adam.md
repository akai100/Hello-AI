```python3
torch.optim.Adam(params: ParamT,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 amsgrad: bool = False,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False,
                 fused: Optional[bool] = None,
                 decoupled_weight_decay: bool = False,
```

+ lr

  初始学习率，默认0.001

+ betas

  一阶/二阶矩的指数衰减系数，默认(0.9, 0.999)

+ eps

  数值稳定项，避免分母为0，默认1e-8

+ weight_decay

  权重衰减（L2正则化），默认0

+ amsgrad

  是否使用AMSGrad变体，默认False

+ foreach

  是否使用foreach实现，默认None

+ maximize

  是否最大化目标函数，默认False

+ capturable

  是否支持CUDA图捕获，默认False

+ differentiable

  是否支持求导，默认False

+ fused

  是否使用融合实现加速，默认None

## 核心原理

1. 一阶矩（动量）更新：累积梯度的指数移动平均（对应动量）

 $m_t=\beta_1 \cdot m_{t-1}+(1-\beta_1)\cdot g_t$

2. 二阶矩（梯度平方）更新：累积梯度平方的指数移动平均（用于自适应学习率）

 $v_t=\beta_2 \cdot v_{t-1}+(1-\beta_2)\cdot g_t^2$

3. 偏差修正：由于初始 $m_0 = 0、v_0 = 0$，前几轮的 $m_t$和 $v_t$会偏小，需修正：

 $\hat{m_t}=\frac{m_t}{1-\beta_{1}^t}, \hat{v_t}=\frac{v_t}{1-\beta_{2}^{t}}$

4. 参数更新：结合修正后的一阶矩 / 二阶矩，计算自适应步长：

 $\beta_{t+1}=\theta_t - \eta \cdot \frac{\hat{m_t}}{\sqrt{\hat{v_t}}+ε} + \lambda \cdot \theta_t$

（注：$\lambda$是权重衰减系数，无权重衰减时最后一项为 0）

直观理解：不仅带惯性，还能 “根据路况调整步长”—— 梯度大的参数（路况差）走小步，梯度小的参数（路况好）走大步，最终所有参数都能平稳到达最优值。
