# 任务4：条件注入

## 核心问题
cross attention，拼接求和，自回归中放前面固定，三种方案

## 三种条件注入方案

### 方案1：Cross Attention
```python
# 伪代码
query = target_features
key = condition_features
value = condition_features
output = cross_attention(query, key, value)
```

**特点**：
- 动态交互
- 灵活性高
- 计算开销较大

### 方案2：拼接求和
```python
# 伪代码
combined = concat([target_features, condition_features], dim=-1)
# 或
combined = target_features + condition_features
```

**特点**：
- 简单直接
- 计算高效
- 可能丢失细节

### 方案3：自回归中放前面固定
```python
# 伪代码
sequence = [condition_tokens, target_tokens]
output = autoregressive_model(sequence)
```

**特点**：
- 强制前置条件
- 生成时顺序依赖
- 适合序列生成

## 参考方向

- 对比不同注入方式效果
- 根据具体任务选择方案
