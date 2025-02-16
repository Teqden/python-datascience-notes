# Numpy
NumPy（Numerical Python）是 Python 的数值计算库，用于 高效地处理大规模数据。它的核心是 ndarray（N 维数组），支持向量化运算，比 Python 自带的 list 快得多。

## 1. 导入并查看版本
```python
# 导入numpy库
import numpy as np
# 查看numpy版本
np.__version__  # 输出结果 '1.26.4'
```

## 2. 创建数组
NumPy 的 ndarray 具有同质性（Homogeneous），数组中的所有元素必须是相同的数据类型（dtype）。如果传入的列表包含不同的数据类型，NumPy 会自动统一为同一种类型，优先级为：`字符串 > 浮点数 > 整数`。

### 2.1 普通方式
#### 2.1.1 np.array()
```python
arr = np.array([1, 2, 3, 4])
print(arr)  # 输出结果 [1 2 3 4]
```

### 2.2 快捷方式
NumPy 提供了一些快捷创建数组的方法（routines 函数），可以快速初始化特定形状的数组，比如全 1、全 0、特定值、单位矩阵等。
#### 2.2.1 np.ones(shape, dtype=None, order='C')
创建一个所有元素都是 1 的数组。
- `shape`：指定数组的形状，可以是整数或元组
    - shape = (m, n) `m 行 n 列` 二维数组
    - shape = (m, 1) `m 行 1 列` 二维数组 `[[1], [2], [3]]`
    - shape = (1, n) `1 行 m 列` 二维数组 `[[1, 2, 3]]`
- `dtype`：元素的数据类型（默认 `float64`）
- `order`：存储顺序（`C` 代表行优先，`F` 代表列优先）
```python
# 创建一个 3x3 的全 1 数组
arr = np.ones((3, 3))
print(arr)

# 输出结果：
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
```

#### 2.2.2 np.zeros(shape, dtype=float, order='C')
创建一个所有元素都是 0 的数组。
```python
# 创建 2x4 的整数 0 数组
arr = np.zeros((2, 4), dtype=int)
print(arr)

# 输出结果：
[[0 0 0 0]
 [0 0 0 0]]
```

#### 2.2.3 np.full(shape, fill_value, dtype=None, order='C')
创建一个所有元素都是 `fill_value` 的数组。
```python
# 创建一个 3x3 的全 7 数组
arr = np.full((3, 3), 7)
print(arr)

# 输出结果：
[[7 7 7]
 [7 7 7]
 [7 7 7]]
```

#### 2.2.4 np.eye(N, M=None, k=0, dtype=float)
创建单位矩阵（对角线为 1，其他位置为 0）。
- `N`：行数
- `M`：列数（可选，默认等于 `N`）
- `k`：对角线偏移（0 是主对角线，+1 上移一行，-1 下移一行）
- `dtype`：数据类型（默认 `float`）
```python
# 3x3 单位矩阵
arr = np.eye(3)
print(arr)

# 输出结果：
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

# 对角线向上偏移 1
arr = np.eye(4, k=1)  
print(arr)

# 输出结果：
[[0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]
 [0. 0. 0. 0.]]
```

#### 2.2.5 np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
在指定范围 `[start, stop]` 内，生成 `num` 个等间距的数，做等差数列。
- `start`：起始值
- `stop`：终止值
- `num`：生成的元素个数（默认 50）
- `endpoint`：是否包含 `stop`（默认 `True`，包含）
- `retstep`：是否返回步长 step（默认 `False`）
- `dtype`：数据类型
```python
# 在 1 和 10 之间取 5 个数
arr = np.linspace(1, 10, 5)
print(arr)
# 输出结果：[ 1.    3.25  5.5   7.75 10.  ]
```

#### 2.2.6 np.arange(start, stop, step, dtype=None)
在 `[start, stop)` 区间内，以 `step` 为步长生成等间距的数（类似 `range()`）。
- `start`：起始值（默认 0）
- `stop`：终止值（不包含 `stop`）
- `step`：步长（默认 1）
- `dtype`：数据类型（可选）
```python
# 从 1 到 9（不含 10），步长为 2
arr = np.arange(1, 10, 2)
print(arr) # 输出结果：[1 3 5 7 9]
```

#### 2.2.7 np.random.randint(low, high=None, size=None, dtype=int)
生成 `size` 个 `low` 到 `high-1` 之间的随机整数。
- `low`：最小值（包含）
- `high`：最大值（不包含，默认 `None` 时，取 0 到 `low-1`）
- `size`：生成的数组形状（可选，默认 `None`，即返回一个数），和之前其他函数的 `shape` 差不多
- `dtype`：数据类型（默认 `int`）
```python
# 生成一个 1~9 之间的随机整数
rand_num = np.random.randint(1, 10)
print(rand_num) # 输出结果：7  （每次运行结果可能不同）

# 生成多个随机整数，并且是一个3x3 矩阵
arr = np.random.randint(1, 10, size=(3, 3))
print(arr)

# 输出结果（每次运行结果可能不同）
[[2 8 5]
 [3 7 1]
 [6 9 4]]
```

#### 2.2.8 np.random.randn(d0, d1, ..., dn)
生成符合 标准正态分布（均值 0，标准差 1）的随机数。
- `d0, d1, ..., dn`：生成的数组形状（可选）；如果不传参数，则返回一个单独的数；如果传多个参数，则生成对应形状的数组
```python
# 生成一个符合标准正态分布的随机数
num = np.random.randn()
print(num)  # 输出结果：-0.5123  （每次运行可能不同）

# 生成多个随机数，并且是一个 3x3 的矩阵
arr = np.random.randn(3, 3)
print(arr)
# 输出结果（每次运行可能不同）：
[[-0.89  1.23  0.34]
 [-0.56 -0.78  0.12]
 [ 1.45 -0.23  0.98]]
```

#### 2.2.9 np.random.normal(loc=0.0, scale=1.0, size=None)
普通正态分布，可以指定均值和标准差。
- `loc`：均值（默认 0.0）
- `scale`：标准差（默认 1.0）
- `size`：生成的形状（可以是整数或元组，如 `(3, 3)`）
```python
arr = np.random.normal(loc=5, scale=2, size=(3, 3))  # 均值 5，标准差 2
print(arr)
# 输出结果：
[[5.23  2.18  6.45]
 [3.56  5.98  7.12]
 [4.67  6.78  3.89]]
```

#### 2.2.10 np.random.random(size=None)
均匀分布，生成 0 到 1 之间的随机数，所有数的概率相同。
- `size`：生成的形状（可选，默认 `None`，即返回一个数）
```python
# 生成一个 [0,1) 之间的随机数
num = np.random.random()
print(num) # 输出结果：0.7382  （每次不同）

# 生成多个随机数，并且是一个 3x3 的随机数组
arr = np.random.random((3, 3))
print(arr)
# 输出结果（每次运行可能不同）：
[[0.12 0.98 0.45]
 [0.23 0.76 0.34]
 [0.87 0.65 0.89]]
```
