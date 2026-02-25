#!/bin/bash
# 训练脚本入口（提交规范建议）
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "$SCRIPT_DIR/train"
python train.py
