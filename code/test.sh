#!/bin/bash
# 预测脚本入口（提交规范要求）
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "$SCRIPT_DIR/test"
python predict.py
