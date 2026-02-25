# AI 街景字符编码识别

## 1. 环境依赖
- 操作系统：Windows / Linux
- Python：3.12.4
- PyTorch：2.7.1（`torchvision` 0.22.1）

安装依赖：
```bash
pip install -r requirements.txt
```

## 2. 方案简介
本方案使用定长多字符分类（最长 6 位）：
- Backbone：预训练 ResNet18 提取图像特征。
- Head：6 个分类头分别预测 6 个位置。
- 类别定义：`0-9` 为数字，`10` 为空白字符。

## 3. 提交目录结构（规范）
```text
project/
├── README.md
├── requirements.txt
├── tcdata/                      # 评测环境会注入原始数据（无需提交数据文件）
├── user_data/
│   ├── model_data/
│   │   └── best_model.pth       # 训练后模型权重
│   └── tmp_data/
├── prediction_result/
│   └── result.tsv               # 预测输出文件
└── code/
    ├── train/
    │   ├── train.py
    │   ├── dataset.py
    │   ├── model.py
    │   └── ...
    ├── test/
    │   └── predict.py
    ├── train.sh                 # 训练入口
    └── test.sh                  # 预测入口（必选）
```

## 4. 固定训练超参数（当前版本）
位于 `code/train/train.py`：
- 输入尺寸：`64x128`
- Batch Size：`256`
- Epoch：`20`
- 损失函数：`CrossEntropyLoss(label_smoothing=0.1)`
- 优化器：`Adam(lr=0.001, weight_decay=5e-5)`
- 学习率策略：`StepLR(step_size=4, gamma=0.5)`

说明：代码中保留了 `FocalLoss`（含 `label_smoothing`）实现，可用于消融实验，但当前默认训练配置为上面的 CE 版本。

## 5. 训练流程
1. 准备数据到 `tcdata/`（评测环境会自动放入）。
2. 执行训练入口：
```bash
bash code/train.sh
```

训练产物：
- 模型：`user_data/model_data/best_model.pth`
- 日志：`logs/train_history.csv`、`logs/best.txt`、`logs/tb/`

## 6. 预测流程
1. 确保 `user_data/model_data/best_model.pth` 已存在。
2. 执行预测入口：
```bash
bash code/test.sh
```
3. 输出文件：`prediction_result/result.tsv`

`result.tsv` 格式：
```text
file_name<TAB>file_code
000000.png<TAB>123
```

## 7. 运行命令（Windows 可选）
若本地无 `bash`，可直接用 Python 执行：
```bash
python code/train/train.py
python code/test/predict.py
```

## 8. 复现与路径说明
- 训练与预测代码均使用相对路径定位 `tcdata/`、`user_data/`、`prediction_result/`。
- 根目录 `train.sh`、`test.sh` 为兼容入口，会转发到 `code/train.sh`、`code/test.sh`。

## 9. 提交前检查
- `code/test.sh` 可执行并生成 `prediction_result/result.tsv`
- `user_data/model_data/best_model.pth` 已存在
- 提交包内包含：`README.md`、`requirements.txt`、`code/`、`user_data/`、`prediction_result/`
