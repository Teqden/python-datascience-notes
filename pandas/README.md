# Pandas
NumPy：提供高效的数值计算，适用于数组（ndarray）运算<br>
Pandas：提供数据分析和业务逻辑处理，适用于表格数据（DataFrame）

## 1. 导入
```python
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
```

## 2. Series
Series 是 Pandas 提供的一种数据结构，类似一维数组，但比 NumPy 的 ndarray 更强大，因为它支持索引（index），可以像字典一样存取数据。
- `values`：存储数据（NumPy ndarray 类型）
- `index`：存储索引（可以是数字、字符串等）

### 2.1 创建Series
#### 2.1.1 由列表创建
使用 `pd.Series()` 从列表或 NumPy 数组创建 Series，默认索引是 0 到 n-1 的整数型索引。
- Series 默认的索引是 0, 1, 2, 3，类似 `range(len(data))`
- `values` 存储 NumPy ndarray 数据，但不会影响原列表（因为 list 是可变对象）
```python
names = ["tom", "lucy", "jack". "maria"]
s = Series(names)
print(s) 
# 输出结果：
0 tom
1 lucy
2 jack
3 maria
dtype: object
```

如果需要自定义索引，可以使用 `index` 参数。
- `index` 可以是自定义的标签，类似字典的键
```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s)
# 输出结果：
a    10
b    20
c    30
d    40
dtype: int64
```
#### 2.1.2 由numpy数组创建
默认索引仍然是 0, 1, 2, 3。
```python
arr = np.array([1, 2, 3, 4])
s = pd.Series(arr)
print(s)
# 输出结果：
0    1
1    2
2    3
3    4
dtype: int64
```
Series 不会复制 NumPy ndarray 的数据，而是引用原数组的数据。修改 Series 的元素时，原 ndarray 也会改变。
```python
arr = np.array([1, 2, 3, 4])
s = pd.Series(arr)
s[0] = 100 # 修改 Series 中的第一个元素

print(s) 
# 输出结果：
0    100
1      2
2      3
3      4
dtype: int64

print(arr) # 输出结果：[100   2   3   4]
```

#### 2.1.3 由字典创建
使用字典（dict）创建 Series，它会自动使用字典的键作为索引，值作为 Series 的数据（`values`）。
- Series 的索引来自字典的键（'apple', 'banana', 'cherry'）。
- values 直接来自字典的值（5, 10, 15）。
```python
data = {'apple': 5, 'banana': 10, 'cherry': 15}  
s = pd.Series(data)  
print(s)
# 输出结果：
apple      5
banana    10
cherry    15
dtype: int64
```

如果提供 index 参数，Series 会按照 index 指定的顺序排列数据，并且：
- 如果 index 中的某个键不存在于字典中，它的值会是 `NaN`（pandas 使用 `NaN` 表示缺失值）。
- 如果 index 少于字典中的键，Series 只会保留 index 指定的部分。
```python
keys = ['banana', 'apple', 'grape']  # 指定顺序，并加入一个不存在的键 'grape'
s = pd.Series(data, index=keys)
print(s)
# 输出结果：
banana    10.0
apple      5.0 # banana 和 apple 按照 index 指定的顺序排列。
grape      NaN # grape 在原字典中不存在，因此变成 NaN（缺失值）。
dtype: float64 # dtype 变成 float64，因为 NaN 只能和浮点数兼容。
```

### 2.2 访问机制
```python
data = {'apple': 5, 'banana': 10, 'cherry': 15}
s = pd.Series(data)
```
#### 2.2.1 loc[]（使用显式索引）
用 `loc[]` 访问数据时，必须提供 Series 的 index 标签（dict 的 key）。只能使用 index 的值 作为索引，不能用位置索引。
```python
print(s.loc['apple'])  # 获取 'apple' 的值
print(s.loc[['apple', 'cherry']])  # 获取多个值
# 输出结果：
5
apple      5
cherry    15
dtype: int64
```

#### 2.2.2 iloc[]（使用隐式索引）
用 `iloc[]` 访问数据时，只能使用 0, 1, 2, ... 这样的整数索引，类似 numpy 数组的索引。`iloc[]` 对应的是 Series 中元素的位置，与 index 无关。
```python
print(s.iloc[0])  # 获取第 0 个元素
print(s.iloc[[0, 2]])  # 获取第 0 和 第 2 个元素
# 输出结果：
5
apple      5
cherry    15
dtype: int64
```

#### 2.2.3 使用布尔索引访问 Series
使用 `bool` 数组筛选数据
```python
s = pd.Series([10, 20, 30, 40, 50])
print(s[s > 25])  # 选取大于 25 的元素
# 输出结果：
2    30
3    40
4    50
dtype: int64
```

### 2.3 基本属性
```python
s = pd.Series([10, 20, 30, 40, 50])
```
- `shape`：返回 Series 的形状（即 `(n,)`，n 表示元素个数），适用于多维数据。`print(s.shape)`的输出结果为：`(5,)`。
- `size`：返回 Series 的元素总个数，不受 `NaN` 影响。`print(s.size)`的输出结果为：`5`。
- `index`：返回 Series 的索引（类似 list 或 ndarray 的 index）。`print(s.index)`的输出结果为：`RangeIndex(start=0, stop=5, step=1)`；还可以通过调用`s.index = ['a', 'b', 'c', 'd', 'e']`来修改索引。
- `values`：返回 Series 的值，类型是 ndarray。`print(s.values)`的输出结果为：`[10 20 30 40 50]`。
- `name`：Series 可以有自己的名称，用于标识数据的含义。
    - ```python
        s.name = "Sales"
        print(s)
        # 输出结果：
        0    10
        1    20
        2    30
        3    40
        4    50
        Name: Sales, dtype: int64
        ```

### 2.4 预览数据
- `head(n)`：返回前 n 个元素，比如 `s.head(3)` 指取前 3 个。
- `tail(n)`：返回后 n 个元素，比如 `s.tail(2)` 指取后 2 个。

### 2.5 处理缺失值
```python
s = pd.Series([1, 2, None, 4, 5])
```
`pd.isnull(s)` / `s.isnull()`：判断 Series 中哪些值是 `NaN`（缺失值）。
```python
print(pd.isnull(s)) # 也可以用 s.isnull()
# 输出结果：
0    False
1    False
2     True
3    False
4    False
dtype: bool
```

`pd.notnull(s)` / `s.notnull()`：判断 Series 中哪些值不是 `NaN`。
```python
print(pd.notnull(s))  # 也可以用 s.notnull()
# 输出结果：
0     True
1     True
2    False
3     True
4     True
dtype: bool
```

### 2.6 排序数据及统计值出现的次数
#### 2.6.1 按值排序
```python
s = pd.Series([30, 10, 20, 50, 40])
```
升序排序：`s.sort_values()`<br>
降序排序：`s.sort_values(ascending=False)`

#### 2.6.2 按索引排序
```python
s.index = ['c', 'a', 'b', 'e', 'd']
```
按索引升序排序：`s.sort_index()`

#### 2.6.3 统计值出现的次数
`value_counts()`：统计 Series 中各个值的出现次数。
```python
s = pd.Series(['apple', 'banana', 'apple', 'cherry', 'banana', 'banana'])
print(s.value_counts())
# 输出结果：
banana    3
apple     2
cherry    1
dtype: int64
```