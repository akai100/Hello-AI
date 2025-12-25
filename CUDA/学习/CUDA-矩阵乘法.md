## 1. 最简单的矩阵乘法

```c++
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
           tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
```

调用核函数：

```c++
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
dim3 blockDim(32, 32, 1);

sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

### 1.1 计算时间分析

对于两个 $4092^2$ 矩阵的矩阵乘法，随后加上一个 $4092^2$ 矩阵：

+ 总浮点运算次数： $2*4092^3+4092^2 137 GFLOPS$

+ 需要读取的总数据量（最小值）： $3 * 4092^2 * 4B = 201MB$

+ 存储的总数据： $4092^2 * 4b = 67MB$

### 1.2 简单核函数的内存访问模式

在我们的核函数中，同一线程中线程 ID 为 （0， 0）和（0，1）的两个线程将加载 B 的同一列，但加载 A 的不同行。如果我们假设缓存为零
的最坏情况，那么每个线程必须从全局内存加载 $2*4092+1$ 个浮点数。 由于我们总共有 $4092^2$ 个线程，这将导致 548 GB 的内存流量。

## 2. 内核 2：全局内存合并

wrap，属于同一线程束的线程进行的连续内存访问可被组合起来并作为一个整体执行。这被称为全局内存合并。

每个多处理器有四个 wrap 调度器，wrap 的分组基于 ```threadId```进行。

```c++
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}
```

调用

```c++
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));

dim3 blockDim(32 * 32);
sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C)
```

## 3. 内核3：共享内存缓存分块

每个流式多核处理器（SM）都有一个共享内存。

由于共享内存位于芯片上，因此与全局内存相比，它具有更低的延迟和更高的带宽。

因此，对于下一个核函数的优化点，我们将从全局内存中加载 A 的一个快和 B 的一个快到共享内存中。

然后，我们将对这两个快执行尽可能多的操作，每个线程仍然分配给 C 的一个元素。我们将沿着 A 的列
和B的行移动这些快，对C执行部分求和，直到计算出结果。

```C++
A += cRow * BLOCKSIZE * K;
B += cCol * BLOCKSIZE;
C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

float tmp = 0.0;

for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
}
```

blockDim x

## 内核4：用于每个线程计算多个结果的一维快分块

## 内核 5：通过二维快分块提高算术强度

## 内核6：向量化共享内存（SMEM）和全局内存（GMEM）访问

## 9. 内核9：自动调优

## 10. 内核10：warp 分块

## 11. 内核11：Kernel 11 进行中
