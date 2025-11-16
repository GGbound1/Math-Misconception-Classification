# 数学误解分类项目

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

基于机器学习的学生数学误解识别系统。

## 🏆 竞赛

- **竞赛名称**: [MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)
- **参与形式**: 个人参赛

## 🚀 快速开始

### 环境安装
```bash
git clone https://github.com/G6bound1/Math-Misconception-Classification.git
cd Math-Misconception-Classification
pip install -r requirements.txt
模型训练
bash
python src/train.py
生成预测
bash
python src/predict.py
🛠 技术方案
特征工程
文本统计特征: 词数、字符数、词汇多样性、句子结构

数学特征: 数学术语频率、运算符计数、分数/小数检测

语言模式: 疑问词、正负面情感指示器

TF-IDF特征: 词级别和字符级别的n-gram

模型架构
集成方法: 软投票分类器，带权重概率

基础模型:

LightGBM (权重: 3) - 优化处理类别不平衡

XGBoost (权重: 2) - 在结构化数据上表现强劲

随机森林 (权重: 1) - 提供预测多样性

流水线: 集成的特征工程和模型训练

高级技术
5折分层交叉验证

类别不平衡的权重调整

基于概率的前3预测排序

全面的文本预处理和清洗

📁 项目结构
text
数学-误解-分类/
├── src/
│   ├── data_loader.py          # 数据加载和预处理
│   ├── feature_engineering.py  # 高级特征提取
│   ├── model.py               # 集成模型定义
│   ├── train.py               # 模型训练脚本
│   └── predict.py             # 预测生成脚本
├── notebooks/                 # 探索性分析笔记本
├── models/                    # 保存的模型文件
├── data/                      # 数据目录
├── outputs/                   # 预测输出
├── requirements.txt           # Python依赖包
└── README.md                 # 项目文档
🔧 核心功能
数据加载器 (data_loader.py)
多源数据高效加载

文本预处理和缺失值处理

类别到误解的映射建立

特征工程 (feature_engineering.py)
自定义语言特征转换器

数学内容检测

词级别和字符级别的TF-IDF向量化

模型框架 (model.py)
可配置权重的集成分类器

支持多种机器学习算法

基于概率的预测排序

📊 模型性能
模型使用5折分层交叉验证进行评估，在从文本解释预测学生数学误解方面表现出稳健的性能。

💡 应用价值
本解决方案可帮助教育工作者：

自动识别常见的学生误解模式

提供针对性的反馈和干预

扩展个性化学习支持

分析数学思维模式

🛠️ 依赖环境
Python 3.7+

scikit-learn

LightGBM

XGBoost

pandas

numpy

joblib

完整版本信息请参见 requirements.txt 文件。

📄 许可证
本项目采用 MIT 许可证 - 详见 LICENSE 文件。

🤝 贡献
欢迎贡献代码！请随时提交 Pull Request。

📧 联系
有关此项目的问题，请在GitHub上提交issue。
