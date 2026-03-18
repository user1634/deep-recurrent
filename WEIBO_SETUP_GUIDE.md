# 微博评论数据集 + DRNT 深度学习配置指南

## 📋 概览

本指南说明如何将 HuggingFace 的 `weibo-comments-v1` 数据集转换为 DRNT (Deep Recurrent Neural Networks) 的输入格式，用于微博情感表达式提取任务。

## 🔄 工作流程

```
HuggingFace 数据集
    ↓
Python 转换脚本
    ↓
DRNT 格式数据
    ↓
编译 + 训练 DRNT 模型
```

## 📥 第一步：转换数据集

### 方式 A：自动从 HuggingFace 下载（推荐）

```bash
# 安装必要的 Python 库
python3 -m venv venv
source venv/bin/activate
pip install datasets

# 运行转换脚本
python3 convert_weibo_dataset.py
```

**脚本会：**
- ✓ 从 HuggingFace 下载 weibo-comments-v1
- ✓ 分词和 BIO 标注（情感表达式识别）
- ✓ 生成 DRNT 格式的数据文件
- ✓ 创建 10-fold 交叉验证分割
- ✓ 保存到 `weibo_drnt_data/` 目录

### 方式 B：手动添加数据文件

如果无法自动下载，手动创建以下文件结构：

```
weibo_drnt_data/
├── weibo_comments.txt       # 主数据文件
├── sentenceid.txt           # 句子-文档映射
└── datasplit/
    ├── doclist.mpqaOriginalSubset
    ├── filelist_train0 ~ filelist_train9
    └── filelist_test0 ~ filelist_test9
```

**weibo_comments.txt 格式：**
```
字\tPOS\tBIO_label
字\tPOS\tBIO_label
...

下一个句子
```

示例：
```
这\tNN\tO
太\tNN\tO
棒\tNN\tB
了\tNN\tI
，\tNN\tO

你\tNN\tO
很\tNN\tO
烂\tNN\tB
```

**标签说明：**
- **B (Begin)**: 情感/观点表达式的开始词
- **I (Inside)**: 情感/观点表达式的继续词
- **O (Outside)**: 非情感词汇

## 🛠️ 第二步：准备词嵌入

### 方式 1：使用已有的嵌入（推荐）

转换脚本会从数据中自动生成词嵌入。如果需要更高质量的嵌入，可以使用预训练的中文词向量：

```bash
# 示例：使用腾讯词向量
# 下载：https://ai.tencent.com/ailab/nlp/zh/pretrain.html

# 转换为 DRNT 格式
python3 convert_embeddings.py tencent_embeddings.txt embeddings-original.EMBEDDING_SIZE=300.txt
```

### 方式 2：使用转换脚本生成的嵌入

脚本自动生成 25 维嵌入，无需额外操作。

## 📦 第三步：编译 DRNT

### 编译微博版本

```bash
# 编译为微博评论专用版本
g++ drnt_weibo.cpp -I ./Eigen/ -std=c++11 -O3 -o drnt_weibo
```

### 或使用原始版本

```bash
# 重命名数据文件以兼容原始 drnt.cpp
mv weibo_drnt_data/weibo_comments.txt ese.txt
g++ drnt.cpp -I ./Eigen/ -std=c++11 -O3 -o drnt
```

## 🚀 第四步：训练模型

### 方式 1：单折训练

```bash
# 训练第 0 折
./drnt_weibo 0

# 输出示例：
# Loaded 12400 sentences
# Train: 8680 | Test: 1860 | Valid: 1860
# Epoch 0
# P, R, F1 (train):
# ...
```

### 方式 2：K-fold 交叉验证（全部 10 折）

```bash
#!/bin/bash
for fold in {0..9}; do
    echo "Training fold $fold..."
    ./drnt_weibo $fold
done
```

### 方式 3：批量运行与结果统计

```bash
python3 << 'EOF'
import subprocess
import re

results = []
for fold in range(10):
    print(f"Running fold {fold}...")
    output = subprocess.check_output(f"./drnt_weibo {fold}", shell=True, text=True)
    # 提取最终结果
    if "Best Results" in output:
        results.append(output)

# 统计平均 F1 分数
print("\n=== Cross-Validation Results ===")
for i, result in enumerate(results):
    print(f"Fold {i}:\n{result}\n")
EOF
```

## 📊 输出说明

训练输出包含：

```
Train metrics (训练集)
├── P (Precision): 精确度 - 预测的情感表达中有多少是正确的
├── R (Recall): 召回率 - 真实情感表达中有多少被找到
└── F1: 精确度和召回率的调和平均数

Valid metrics (验证集)
└── 用于模型选择

Test metrics (测试集)
└── 最终评估指标
```

## 🔧 配置调整

在 `drnt_weibo.cpp` 中修改：

```cpp
#define ETA 0.001          // 学习率
#define DROPOUT            // 启用/禁用 dropout
#define OCLASS_WEIGHT 0.5  // "O" 标签的权重（降低）
#define layers 2           // 额外隐藏层数
#define MR 0.7             // 动量系数

double LAMBDA = 1e-4;      // L2 权重正则化
double LAMBDAH = 1e-4;     // L2 激活正则化
```

### 参数调整建议

| 参数 | 说明 | 调整 |
|------|------|------|
| ETA | 学习率 | 小于 0.001 时收敛慢，大于 0.01 可能不稳定 |
| LAMBDA | 权重衰减 | 增加以防止过拟合 |
| OCLASS_WEIGHT | O 标签权重 | 减小以处理不平衡的 O/B/I |
| layers | 隐藏层数 | 更多层可能捕获更复杂的模式 |

## 📂 输出文件

训练完成后生成：

```
models/
└── drnt_weibo_2_25_25_0.2_200_0.001_0.0001_0.7_0
    └── 训练好的模型权重文件

model.txt
└── 最后保存的模型
```

## 🐛 常见问题

### 1. "embeddings-original.EMBEDDING_SIZE=25.txt not found"

解决方案：
```bash
# 重新运行转换脚本
python3 convert_weibo_dataset.py
```

### 2. "No such file: weibo_drnt_data/weibo_comments.txt"

解决方案：
```bash
# 检查数据转换是否成功
ls -la weibo_drnt_data/
# 重新运行转换脚本
python3 convert_weibo_dataset.py
```

### 3. 训练速度很慢

优化建议：
- 减少 MAXEPOCH（在 drnt_weibo.cpp 中）
- 增加 MINIBATCH 大小
- 检查 CPU 使用率，考虑升级硬件

### 4. F1 分数很低

调试步骤：
- 检查数据是否正确标注（查看 weibo_comments.txt）
- 尝试调整学习率和正则化参数
- 增加隐藏层数或维度
- 准备更多的训练数据

## 📚 数据转换详解

### 转换脚本工作流程

```python
# 1. 加载微博数据
dataset = load_dataset("wsqstar/weibo-comments-v1")

# 2. 分词（使用简单的字符级分词，可改为 jieba）
tokens = list(comment_text)

# 3. 情感词识别和 BIO 标注
labels = []
for token in tokens:
    if token in sentiment_lexicon:
        labels.append("B")  # 或 "I"
    else:
        labels.append("O")

# 4. 保存为 DRNT 格式
for token, label in zip(tokens, labels):
    f.write(f"{token}\tNN\t{label}\n")

# 5. 创建文件夹分割
train_docs = [...]
test_docs = [...]
for fold in range(10):
    save_split(fold, train_docs, test_docs)
```

## 🎯 下一步

1. ✅ 运行 `python3 convert_weibo_dataset.py`
2. ✅ 检查生成的数据文件
3. ✅ 编译 `g++ drnt_weibo.cpp -I ./Eigen/ -std=c++11 -O3 -o drnt_weibo`
4. ✅ 训练 `./drnt_weibo 0`
5. ✅ 评估结果和调整超参数

## 📖 参考文献

- **DRNT 原论文**: Irsoy, O., & Cardie, C. (2014). Opinion Mining with Deep Recurrent Neural Networks. EMNLP.
- **微博评论数据集**: [weibo-comments-v1 on HuggingFace](https://huggingface.co/datasets/wsqstar/weibo-comments-v1)
- **Eigen 库**: [libeigen.org](http://eigen.tuxfamily.org)

## 💡 建议和改进

当前实现使用：
- ✓ 字符级分词（简单）
- ✓ 简单情感词匹配

可以改进为：
- 🔲 使用 jieba 进行词级分词
- 🔲 集成更大的情感词汇表
- 🔲 使用真实的 POS 标签（而非所有 "NN"）
- 🔲 多任务学习（同时预测情感极性）
- 🔲 加入上下文预训练词向量
