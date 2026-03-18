#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Weibo Comments Dataset to DRNT Format

原始格式: JSON/CSV with comments and sentiment labels
目标格式: DRNT format (token\tPOS\tBIO_label)

BIO标签定义:
- B (Begin): 情感/观点表达的开始
- I (Inside): 情感/观点表达的继续
- O (Outside): 非情感/观点词汇
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import random
from typing import List, Tuple, Dict
import re

class WeiboToDRNT:
    def __init__(self, output_dir="weibo_drnt_data"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
        Path(f"{output_dir}/datasplit").mkdir(exist_ok=True)

        self.sentences = []  # 所有句子
        self.labels = []     # 对应的标签序列
        self.doc_ids = []    # 每个句子属于的文档
        self.sentiment_words = self._load_sentiment_lexicon()

    def _load_sentiment_lexicon(self) -> Dict[str, str]:
        """
        加载情感词汇表（可选）
        这里使用简单的规则，可以扩展为使用真实的词汇表
        """
        # 简单的情感词示例
        positive_words = {
            '好': '正', '棒': '正', '喜欢': '正', '爱': '正',
            '不错': '正', '赞': '正', '完美': '正', '优秀': '正'
        }
        negative_words = {
            '差': '负', '烂': '负', '讨厌': '负', '恨': '负',
            '糟糕': '负', '垃圾': '负', '差劲': '负', '令人失望': '负'
        }
        lexicon = {}
        for word, label in positive_words.items():
            lexicon[word] = label
        for word, label in negative_words.items():
            lexicon[word] = label
        return lexicon

    def _segment_and_tag(self, text: str) -> Tuple[List[str], List[str]]:
        """
        分词和标注情感表达
        这里使用简单的字符分词，可以替换为更好的分词器（jieba等）
        """
        # 简单的中文分词（基于字符）
        # 实际应用中建议使用 jieba 或其他专业分词器
        tokens = list(text)  # 这里使用字符级别
        labels = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # 检查是否为情感词
            is_sentiment = False
            for word, sentiment in self.sentiment_words.items():
                if i + len(word) <= len(tokens):
                    matched = ''.join(tokens[i:i+len(word)])
                    if matched == word:
                        # 这是一个情感词，用B标记
                        labels.append('B')
                        for _ in range(1, len(word)):
                            if i + 1 < len(tokens):
                                labels.append('I')
                        is_sentiment = True
                        i += len(word)
                        break

            if not is_sentiment:
                labels.append('O')
                i += 1

        return tokens, labels

    def process_from_huggingface(self):
        """从 HuggingFace 加载并处理数据集"""
        try:
            from datasets import load_dataset
            print("Loading weibo-comments-v1 from HuggingFace...")
            dataset = load_dataset("wsqstar/weibo-comments-v1")

            doc_id = 0
            for split in ['train', 'test']:
                if split not in dataset:
                    continue

                print(f"\nProcessing {split} split ({len(dataset[split])} examples)...")

                for idx, example in enumerate(dataset[split]):
                    if idx % 1000 == 0:
                        print(f"  Processed {idx}/{len(dataset[split])}")

                    # 提取评论文本
                    comment_text = example.get('content', '')

                    if not comment_text or len(comment_text) < 2:
                        continue

                    # 分词和标注
                    tokens, labels = self._segment_and_tag(comment_text)

                    if len(tokens) > 0:
                        self.sentences.append(tokens)
                        self.labels.append(labels)
                        self.doc_ids.append(doc_id)
                        doc_id += 1

            print(f"\nLoaded {len(self.sentences)} sentences from {doc_id} documents")

        except ImportError:
            print("HuggingFace datasets not available. Using demo data instead.")
            self.process_demo_data()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.process_demo_data()

    def process_demo_data(self):
        """使用演示数据（如果无法加载真实数据集）"""
        print("\nUsing demo Weibo comments data...")

        demo_comments = [
            ("这部电影太棒了，我很喜欢这个演员的表演", "positive"),
            ("这个产品质量太差了，非常失望", "negative"),
            ("一般般，不错的电影", "neutral"),
            ("太烂了，浪费了我的时间和金钱", "negative"),
            ("完美的服务，强烈推荐", "positive"),
            ("这个商品没有想象中那么好", "negative"),
            ("爱上这个品牌了，质量很好", "positive"),
        ]

        for idx, (text, sentiment) in enumerate(demo_comments):
            tokens, labels = self._segment_and_tag(text)
            if len(tokens) > 0:
                self.sentences.append(tokens)
                self.labels.append(labels)
                self.doc_ids.append(idx)

    def save_drnt_format(self):
        """保存为 DRNT 格式"""
        # 保存主数据文件
        output_file = f"{self.output_dir}/weibo_comments.txt"
        print(f"\nSaving DRNT format to {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for tokens, labels in zip(self.sentences, self.labels):
                for token, label in zip(tokens, labels):
                    # 格式: token\tPOS\tBIO_label
                    # POS: 这里简化为所有token都标记为"NN"(名词)
                    # 如果需要真实POS标签，可以使用 jieba.posseg
                    f.write(f"{token}\tNN\t{label}\n")
                # 句子之间用空行分隔
                f.write("\n")

        print(f"✓ Saved {len(self.sentences)} sentences")

    def save_metadata(self):
        """保存元数据"""
        # 保存句子ID和文档ID映射
        with open(f"{self.output_dir}/sentenceid.txt", 'w', encoding='utf-8') as f:
            for sent_idx, doc_id in enumerate(self.doc_ids):
                f.write(f"{sent_idx} {doc_id} doc_{doc_id}\n")

        print(f"✓ Saved sentenceid.txt with {len(self.doc_ids)} entries")

    def split_train_test_valid(self, n_folds=10, train_ratio=0.7, test_ratio=0.15):
        """
        分割数据为 train/test/valid，支持 K-fold 交叉验证
        """
        print(f"\nSplitting data into {n_folds} folds...")

        n_docs = max(self.doc_ids) + 1
        doc_list = list(range(n_docs))
        random.seed(42)
        random.shuffle(doc_list)

        fold_size = n_docs // n_folds

        for fold in range(n_folds):
            # 确定这一折的 test 集合文档
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n_docs
            test_docs = set(doc_list[test_start:test_end])

            # 剩余文档用于 train+valid
            remaining_docs = [d for d in doc_list if d not in test_docs]
            split_idx = int(len(remaining_docs) * (train_ratio / (train_ratio + (1 - train_ratio - test_ratio))))

            train_docs = set(remaining_docs[:split_idx])
            valid_docs = set(remaining_docs[split_idx:])

            # 保存文件列表
            with open(f"{self.output_dir}/datasplit/filelist_train{fold}", 'w') as f:
                for doc in sorted(train_docs):
                    f.write(f"doc_{doc}\n")

            with open(f"{self.output_dir}/datasplit/filelist_test{fold}", 'w') as f:
                for doc in sorted(test_docs):
                    f.write(f"doc_{doc}\n")

        # 保存文档列表
        with open(f"{self.output_dir}/datasplit/doclist.mpqaOriginalSubset", 'w') as f:
            for doc in sorted(doc_list):
                f.write(f"doc_{doc}\n")

        print(f"✓ Created {n_folds}-fold cross-validation split")
        print(f"  Documents per fold: ~{fold_size}")
        print(f"  Train/Test/Valid ratio: {train_ratio:.1%}/{test_ratio:.1%}/{(1-train_ratio-test_ratio):.1%}")

    def run(self):
        """执行完整的转换流程"""
        print("=" * 60)
        print("Converting Weibo Comments to DRNT Format")
        print("=" * 60)

        self.process_from_huggingface()
        self.save_drnt_format()
        self.save_metadata()
        self.split_train_test_valid(n_folds=10)

        print("\n" + "=" * 60)
        print("✓ Conversion complete!")
        print("=" * 60)
        print(f"\nOutput directory: {self.output_dir}/")
        print(f"Total sentences: {len(self.sentences)}")
        print(f"Total documents: {max(self.doc_ids) + 1 if self.doc_ids else 0}")
        print("\nFiles generated:")
        print(f"  - weibo_comments.txt (main data)")
        print(f"  - sentenceid.txt (sentence-to-document mapping)")
        print(f"  - datasplit/ (fold-based splits)")
        print("\nNext steps:")
        print("  1. Review the data: cat weibo_drnt_data/weibo_comments.txt | head -20")
        print("  2. Compile DRNT for this dataset")
        print("  3. Train: ./drnt_weibo 0")


if __name__ == "__main__":
    converter = WeiboToDRNT(output_dir="weibo_drnt_data")
    converter.run()
