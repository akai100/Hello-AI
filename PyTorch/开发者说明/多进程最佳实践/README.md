
```torch.multiprocessing```是Python的```multiprocessing```模块的即插即用替代品。它支持完全相同的操作，但对其进行了扩展，
因此所有通过```multiprocessing.Queue```发送的张量，其数据都会被移入共享内存，并且只会向另一个进程发送一个句柄。

NOTE 注意
```
当一个Tensor被发送到另一个进程时，Tensor的数据会被共享。如果torch.Tensor.grad不为None，它也会被共享。
在没有torch.Tensor.grad字段的Tensor被发送到另一个进程后，它会创建一个特定于进程的标准.gradTensor，
与Tensor的数据已被共享的情况不同，该张量不会在所有进程间自动共享。
```

这使得可以实现各种训练方法，例如Hogwild、A3C或任何其他需要异步操作的方法。
