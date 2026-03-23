# Qwen3.5 VLN-CE 导航测试

测试 Qwen3.5 模型在 VLN-CE 任务上的导航能力（无需训练）

## 快速开始

```bash
# 基本用法
python test_qwen_vln.py --config path/to/config.yaml --model-path path/to/qwen3.5

# 指定输出目录
python test_qwen_vln.py --config configs/test.yaml --model-path ./qwen3.5 --output-dir ./results
```

## 文件说明

- `test_qwen_vln.py` - 主测试脚本
- `qwen_agent.py` - Qwen3.5 智能体实现（待完善）
- `utils.py` - 工具函数
- `README.md` - 本文档

## 待实现功能

1. **QwenNavigationAgent.act()** - 实现 Qwen3.5 的推理逻辑
2. **test_navigation()** - 完善环境初始化和测试循环
3. 添加结果保存和可视化功能

## 参数说明

- `--config`: VLN-CE 配置文件路径（必需）
- `--model-path`: Qwen3.5 模型路径
- `--output-dir`: 结果保存目录（默认: ./results）
