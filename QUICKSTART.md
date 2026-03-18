# 🚀 微博 + DRNT 快速开始指南

## 三步启动

### ✅ 步骤 1：转换微博数据集

```bash
python3 convert_weibo_dataset.py
```

**输出：**
```
✓ Conversion complete!
Output directory: weibo_drnt_data/
Total sentences: XXXX
Files generated:
  - weibo_comments.txt (主数据文件)
  - sentenceid.txt (映射文件)
  - datasplit/ (训练/测试分割)
```

### ✅ 步骤 2：编译 DRNT 微博版本

```bash
g++ drnt_weibo.cpp -I ./Eigen/ -std=c++11 -O3 -o drnt_weibo
```

### ✅ 步骤 3：开始训练

```bash
# 训练单个折
./drnt_weibo 0

# 或运行所有 10 折交叉验证
for fold in {0..9}; do ./drnt_weibo $fold; done
```

---

## 📊 期望的输出

```
DRNT for Weibo Comments - Fold 0
Loaded 12400 sentences
Train: 8680 | Test: 1860 | Valid: 1860

Epoch 0
P, R, F1 (train):
0.xxx  0.xxx
0.xxx  0.xxx
0.xxx  0.xxx

Epoch 5
P, R, F1 (train):
0.xxx  0.xxx
0.xxx  0.xxx
0.xxx  0.xxx
...

=== Best Results ===
Dropout: 0.2
0.xxx  0.xxx
0.xxx  0.xxx
0.xxx  0.xxx
```

---

## 🔧 使用真实数据集

目前脚本使用演示数据。如果要使用真实的微博评论数据集：

### 方法 1：自动下载（推荐）

```bash
# 安装必要库
python3 -m venv venv
source venv/bin/activate
pip install datasets

# 运行转换（会自动下载）
python3 convert_weibo_dataset.py
```

### 方法 2：手动提供数据

编辑 `convert_weibo_dataset.py` 的 `process_demo_data()` 函数，替换为你的数据源。

---

## 📁 文件结构

```
deep-recurrent/
├── drnt.cpp                    # 原始 DRNT（MPQA 数据）
├── drnt_weibo.cpp              # 微博版本 DRNT ✨
├── utils.cpp                   # 工具库
├── convert_weibo_dataset.py    # 数据转换脚本 ✨
├── WEIBO_SETUP_GUIDE.md        # 详细指南
├── QUICKSTART.md               # 本文件
├── Eigen/                      # 线性代数库
├── embeddings-original.EMBEDDING_SIZE=25.txt  # 词向量
└── weibo_drnt_data/            # 生成的微博数据
    ├── weibo_comments.txt
    ├── sentenceid.txt
    └── datasplit/
```

---

## 🎯 关键差异：MPQA vs Weibo

| 特性 | MPQA | 微博 |
|------|------|------|
| **数据源** | 新闻评论 | 微博评论 |
| **任务** | 观点表达式提取 | 情感表达式提取 |
| **文件** | ese.txt | weibo_comments.txt |
| **二进制** | drnt | drnt_weibo |
| **BIO标签** | 观点范围 | 情感词范围 |

---

## 💡 关键参数调整

在 `drnt_weibo.cpp` 中修改：

```cpp
#define ETA 0.001        // 学习率：降低=更稳定，升高=更快学习
#define LAMBDA = 1e-4    // 权重衰减：增加=防止过拟合
#define layers 2         // 隐藏层：更多=更强大但更慢
uint MAXEPOCH = 200      // 训练轮数：减少=更快，增加=更好的收敛
```

### 推荐设置

**快速实验：**
```cpp
#define ETA 0.005
uint MAXEPOCH = 50;  // 快速试验
```

**最佳准确度：**
```cpp
#define ETA 0.001
#define LAMBDA = 1e-5
#define layers 3
uint MAXEPOCH = 200;
```

---

## 🔍 检查结果

训练完成后查看模型和结果：

```bash
# 查看生成的模型
ls -lh models/

# 查看最后保存的模型
cat model.txt | head -20

# 运行多折并汇总结果
for fold in {0..9}; do
    echo "=== Fold $fold ==="
    ./drnt_weibo $fold | tail -5
done
```

---

## 🐛 常见问题

**Q: "datasets not available" 错误**
A: 脚本会自动使用演示数据。若要真实数据，运行：
```bash
pip install datasets
python3 convert_weibo_dataset.py  # 重试
```

**Q: 模型准确度很低**
A: 检查数据质量和 BIO 标注。可尝试：
- 增加训练数据
- 调整学习率
- 使用预训练词向量

**Q: 训练很慢**
A: 可以：
- 减少 MAXEPOCH
- 增加 MINIBATCH 大小
- 使用更小的隐藏维度 (nhf/nhb)

---

## 📚 后续阅读

- 📄 详细配置：`WEIBO_SETUP_GUIDE.md`
- 📜 原始文档：`README.md`
- 📖 DRNT 论文：[Irsoy & Cardie 2014](http://aclweb.org/anthology/D14-1080)

---

## ⚡ 一键脚本

```bash
#!/bin/bash
set -e

echo "🚀 Starting Weibo + DRNT Pipeline..."

echo "1️⃣  Converting dataset..."
python3 convert_weibo_dataset.py

echo "2️⃣  Compiling DRNT..."
g++ drnt_weibo.cpp -I ./Eigen/ -std=c++11 -O3 -o drnt_weibo

echo "3️⃣  Training fold 0..."
./drnt_weibo 0

echo "✅ Done! Check output above for results."
```

保存为 `run_weibo.sh` 并执行：
```bash
chmod +x run_weibo.sh
./run_weibo.sh
```

---

**祝你使用愉快！** 🎉
