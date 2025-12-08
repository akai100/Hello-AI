# 1. LBFGS.md

LBFGS（Limited-memory Broyden-Fletcher-Goldfarb-Shanno）是一种拟牛顿法优化器，由 Jorge Nocedal 在 1980 年代提出，是 BFGS 算法的内存高效变体。
它通过近似计算目标函数的 Hessian 矩阵（二阶导数矩阵）来加速收敛，适合处理小规模到中等规模的优化问题（参数数量通常在百万以下）
