# 任务1：训练、推理架构

## 概述

熟悉多卡训练、数据处理、推理等架构。

## 详细需求

筛选出数据集中，房间拐点数量为 8 的房间（一个户型可能多个房间），划分训练集测试集，去训练。用 MLP 作为编码器，解码器。

## 要求

1. **实现多卡训练，挑选参数，能够拉满显卡性能**
   - 优化训练配置
   - 充分利用 GPU 资源
   - 参数调优

2. **wandb记录训练过程，分析损失函数**
   - 集成 wandb 日志
   - 可视化训练指标
   - 分析损失变化趋势

3. **实现测试、推理接口**
   - 模型测试功能
   - 推理 API 接口
   - 结果验证

## 架构示意

```
[输入数据] → [编码器(MLP)] → [中间特征] → [解码器(MLP)] → [输出结果]
```

## 关键点

- 房间级别自注意力编码
- 数据筛选：房间拐点数 = 8
- 训练/测试集划分
- 多卡并行训练优化

## 多卡启动

`trainer.py` 会在 `torchrun` 设置 `WORLD_SIZE` 和 `LOCAL_RANK` 时自动启用
`DistributedDataParallel`。`batch_size` 表示每张 GPU 的 batch size，总 batch size
为 `batch_size * GPU 数量`。

在 `floor-plan` conda 环境中启动两张卡：

```bash
cd assignment1-train-inference-architecture
PYTHONPATH=. conda run -n floor-plan torchrun --standalone --nproc_per_node=2 <your_train_script.py>
```

训练脚本中只应在主进程创建并传入 W&B run；其他进程传入 `run=None`。训练和评估
完成后调用 `cleanup_distributed()` 释放进程组。
