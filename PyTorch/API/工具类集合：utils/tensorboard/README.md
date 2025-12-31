
torch.utils.tensorboard 用于与 TensorBoard 集成，帮助可视化训练过程中的各种信息，如损失函数、精度、图像和模型参数。

## SummaryWriter

SummaryWriter 是 TensorBoard 用于记录训练过程的类。它允许你记录标量、图像、音频、直方图等，并生成 TensorBoard 可视化。


## 启动 TensorBoard

在训练过程中，你可以使用以下命令启动 TensorBoard，以便可视化训练过程：

```bash
tensorboard --logdir=runs
```
