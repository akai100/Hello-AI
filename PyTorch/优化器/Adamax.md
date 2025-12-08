# 1. Adamax

Adamax 是 Adam 优化器的一个变体，由 Kingma 和 Ba 在 2014 年的论文 Adam: A Method for Stochastic Optimization 中提出。
它使用 无穷范数（L∞ 范数） 来更新学习率，相比 Adam 具有更好的稳定性和收敛性，尤其适合处理 稀疏梯度 或 噪声较大 的场景。
