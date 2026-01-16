---
title: "大作业——医疗数据多模型分析与可视化实践"  
collection: portfolio  
type: "Machine Learning"  
permalink: /portfolio/medical-data-multi-model-analysis  
date: 2026-01-17  
excerpt: "通过EDA、多模型训练（逻辑回归/随机森林/SVM）、聚类及SHAP解释，分析医疗数据并提升预测准确性与可解释性。"  
header:  
  teaser: /images/portfolio/medical-data-multi-model-analysis/age_distribution.png  
tags:  
- 医疗数据  
- 多模型分析  
- EDA  
- 模型解释  
- SHAP  
tech_stack:  
- name: Python  
- name: Pandas  
- name: Scikit-learn  
- name: SHAP  
- name: Matplotlib  
- name: Seaborn  
---  

## 项目背景  
本项目针对医疗数据集进行全面分析，包括数据清洗、探索性可视化、多模型训练与评估、无监督聚类及模型可解释性分析。目标是通过多维度分析提升医疗结局预测的准确性，并利用SHAP工具解释模型决策过程，为临床决策提供支持。  

## 核心实现  

### 1. 数据预处理  
```python  
# 缺失值填充（中位数）  
data_imputed = data.fillna(data.median())  
# 划分特征与标签  
X = data_imputed.drop('HOSPITAL_EXPIRE_FLAG', axis=1)  
y = data_imputed['HOSPITAL_EXPIRE_FLAG']  
# 训练集/测试集分割  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
```  

### 2. 多模型训练  
**逻辑回归**：  
```python  
log_reg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)  
log_reg.fit(X_train, y_train)  
```  

**随机森林**：  
```python  
rf = RandomForestClassifier(n_estimators=100, random_state=42)  
rf.fit(X_train, y_train)  
```  

### 3. 模型解释（SHAP）  
```python  
import shap  
# 初始化解释器  
explainer = shap.Explainer(log_reg, X_train)  
shap_values = explainer(X_train)  
# 生成SHAP蜂群图  
shap.summary_plot(shap_values, X_train)  
```  

## 分析结果  
![年龄分布直方图](/images/portfolio/medical-data-multi-model-analysis/age_distribution.png)  
**分析**：年龄分布呈现正态趋势，集中在某一区间。  

![ROC曲线](/images/portfolio/medical-data-multi-model-analysis/roc_curve.png)  
**分析**：逻辑回归模型AUC值为XX，预测性能良好。  

![SHAP蜂群图](/images/portfolio/medical-data-multi-model-analysis/shap_beeswarm.png)  
**分析**：lab_5235_max是影响结局的关键特征，高值增加不良结局风险。  

```
