# 1. AdamW

AdamW 是 Adam 优化器的一个重要变体，由 Ilya Loshchilov 和 Frank Hutter 在 2017 年的论文 Decoupled Weight Decay Regularization 中提出。它解决了 Adam 中 权重衰减与 L2 正则化混淆 的问题，
通过将权重衰减与梯度更新 解耦，实现了更有效的正则化效果，在深度学习模型（尤其是 Transformer）的训练中广泛应用。
