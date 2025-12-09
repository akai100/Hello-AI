# 1. BCELoss.md

二元交叉熵损失（Binary Cross Entropy Loss, BCELoss），衡量模型预测的概率分布与真实标签的交叉熵。

计算公式：

$Loss(y, \hat{y})=-[y\dot log(\hat{y}) + (1-y)\dot log(1-\hat{y})]$
