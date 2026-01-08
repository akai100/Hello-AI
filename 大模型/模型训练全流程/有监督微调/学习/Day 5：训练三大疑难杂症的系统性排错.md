🎯 今日能力目标

到今天结束，你必须能：

**1. 一眼判断 OOM 是哪一类**

**2. 快速定位 NaN / Inf 的根源**

**3. 知道 loss 不降时，先动哪一个旋钮**

**4. 形成一套“训练急救 checklist”**

## 1️⃣ OOM：7 类真实来源（工程必背）

**结论先行：**

OOM ≠ 显存不够

而是 **显存生命周期失控**

### OOM 类型一览（按真实工程频率）

**① Activation 爆炸（第一杀手）**

+ seq_len 过长

+ gradient checkpoint 没开

+ packing 失控

**📌 症状**

+ forward 直接 OOM

+ batch=1 也炸

**② Optimizer state 爆炸**

+ Adam / FP32

+ 全参训练

**📌 症状**

+ backward / step 时 OOM

**③ Grad 累积爆炸**

+ grad_acc 太大

+ 梯度没清干净

**📌 症状**

+ 几 step 后才 OOM

**④ CUDA Fragmentation（隐形杀手）**

+ 频繁 alloc/free

+ 混用大 / 小 tensor

**📌 症状**

+ 理论显存够

+ 实际突然 OOM

**⑤ Communication buffer**

+ ZeRO-3 / FSDP

+ all-gather buffer

**📌 症状**

+ 多卡才 OOM

+ 单卡没事

**⑥ Dataloader / CPU → GPU 泄漏**

+ tensor 没 .to(device)

+ Python list 缓存

**📌 症状**

+ 显存缓慢爬升

**⑦ 混合精度配置错误**

+ FP32 fallback

+ autocast 失效

**📌 症状**

+ 显存异常大

## 2️⃣ NaN / Inf：最快定位法（极实用）

**NaN 的 90% 来源**

1. learning rate 过大

2. loss scale 失控

3. FP16 下 softmax / layernorm

4. bad batch（极端数据）

**工程急救顺序（记住）**

```
1️⃣ 关 AMP，看 NaN 是否消失
2️⃣ 打印 loss / grad norm
3️⃣ 降 lr（×0.1）
4️⃣ 开 grad clipping
5️⃣ 查数据（空 label / 极长样本）
```

## 3️⃣ loss 不收敛：工程排查优先级

不要一上来调模型

**排查顺序（非常重要）**

**① 数据（第一优先级）**

+ response mask 对吗？

+ label shift 对吗？

+ prompt / response 拼接是否错位？

**📌 30% 问题在这**

**② 学习率 & warmup**

+ 没 warmup？

+ lr 太小 / 太大？

**③ LoRA 配置**

+ rank 太小？

+ target_modules 选错？

**④ batch / grad acc**

+ 有效 batch 太小

+ 梯度噪声大

**⑤ 模型是否被冻结错**

+ requires_grad 错误

+ optimizer 包含冻结参数

## 4️⃣ 训练急救 Checklist（你要背下来）

当训练炸了，你按这个顺序做：

```
□ batch_size = 1 跑通？
□ seq_len 减半？
□ 只留 100 条数据能否 overfit？
□ 关 AMP 是否稳定？
□ 只训 LoRA？
□ lr × 0.1？
□ 打印 grad norm？
```

## 🧪 Day 5 实战任务（非常关键）

### ✅ 任务 1（OOM 判断题）

batch_size=1，seq_len=2048，QLoRA，仍然 OOM，最可能是哪一类？你会先改什么？

### ✅ 任务 2（NaN 定位）

训练 50 step 后突然 NaN，你的前三个操作是什么？（按顺序）

### ✅ 任务 3（loss 不降）

loss 长时间不下降，但显存 / 速度都正常，你最先检查哪 2 个点？为什么？

## 🎯 今日验收标准

你能清楚说出：

排错不是“调参”，而是“缩小问题空间”
