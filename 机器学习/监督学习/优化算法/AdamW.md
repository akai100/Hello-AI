AdamW 修正了 Adam 的一个“原则性错误”

## Adam 的 权重衰减之坑

### 1️⃣ 我们本来想做什么？

在训练中加入 **L2 正则化**（权重衰减）：

$$min_\theta L(\theta)+\frac{\lambda}{2}||\theta||^2$$

直觉：
+ 惩罚大权重
+ 提高泛化能力

### 2️⃣ 在 SGD 里，一切都没问题

SGD + L2 正则：

$$\theta_{t+1}=(1-\eta \lambda)\theta_t-\eta∇L(\theta_t)$$

### 3️⃣ 但在 Adam 里，事情坏了 ❌

Adam 通常“照抄”做法：

$$g_t=∇L(\theta_t)+\lambda \theta_t$$

然后送进 Adam：

$$\theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

**❗ 问题本质**

+**权重衰减被当成了梯度的一部分，**
+**被 Adam 的“自适应缩放”扭曲了**

## 为什么这是一个“原则性错误”？

Adam 的更新是：

$$\Delta\theta \propto \frac{g}{\sqrt{v}} $$

参数在一次迭代中的变化量 和 $\frac{g}{\sqrt{v}}$ 成正比

+ 不同参数

  -> 不同 $v$

  -> 不同缩放

👉 结果是：

| 参数   | 权重衰减强度 |
| ---- | ------ |
| 梯度大的 | 衰减小    |
| 梯度小的 | 衰减大    |

📌 L2 正则本应“对所有参数一视同仁”

📌 Adam 却让它“因人而异”

## AdamW 核心思想

+ 权重衰减 ≠ 梯度

+ 不要让 Adam 碰权重衰减

### 1️⃣ 不把权重衰减加进梯度

❌ Adam（错误方式）：

$$g_t=\nabla L(\theta_t)+\lambda \theta_t$$

✅ AdamW（正确方式）：

$$g_t=\nabla L(\theta_t)$$

## 2️⃣ 在参数更新时“单独衰减”

**AdamW 更新公式**

$$\theta_{t+1}=\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}-\eta \lambda \theta_t$$

或写成：

$$\theta_{t+1}=(1-\eta \lambda)\theta_t-\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

**📌 权重衰减是直接对参数做缩放**

## 几何直觉

**Adam 在做什么？**

+ 在“被拉伸/压缩的空间”里走梯度

+ 连正则项也被一起拉伸了

**AdamW 在做什么？**

+ 梯度部分：在 Adam 的自适应空间里走

+ 正则部分：在原始空间里“均匀收缩”

```
Adam:
  [ 梯度 + 正则 ] → 自适应缩放 → 更新

AdamW:
  梯度 → 自适应缩放 → 更新
  权重 → 直接收缩
```

**👉 这是干净的“解耦（decoupled）”**

## 为什么 AdamW 泛化更好？

✔ 权重衰减行为正确

✔ 不依赖梯度历史

✔ 对所有参数施加同等约束

✔ 在大模型中尤为明显（Transformer）

📌 这也是为什么：

BERT / GPT / ViT / LLaMA 默认 AdamW

## Adam vs AdamW 对比总结

| 维度     | Adam  | AdamW     |
| ------ | ----- | --------- |
| 权重衰减方式 | 加进梯度  | **参数级衰减** |
| 衰减是否一致 | ❌ 不一致 | ✅ 一致      |
| 泛化能力   | 一般    | **更好**    |
| 理论一致性  | ❌     | ✅         |
| 实践推荐   | ❌     | ✅         |

## 实践中的关键建议（很重要）

### 1️⃣ 用 AdamW 时：

+ weight_decay ≠ L2

+ 不要手动把 λθ 加进 loss

+ 框架自带 AdamW 就是对的

PyTorch 示例

```
torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)
```

### 2️⃣ 哪些参数不该衰减

通常 **不对以下参数做 weight decay：**

+ bias

+ LayerNorm / BatchNorm 的 weight

📌 Transformer 训练的标准做法

## 一个常见误解澄清

❌ “AdamW = Adam + weight decay”

**✅ AdamW = Adam + 正确的 weight decay**

## 面试一句话（强烈推荐）

AdamW 通过将权重衰减从梯度更新中解耦，避免了 Adam 中 L2 正则被自适应学习率扭曲的问题，使得正则化行为与 SGD 一致，从而显著提升了模型的泛化性能。
