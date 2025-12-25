交叉熵损失函数。等价于 ```nn.LogSoftmax()``` + ```nn.NLLLoss()```，直接接收模型输出的 logits（未经过softmax），自动完成 softmax 归一化和负对数似然计算。
