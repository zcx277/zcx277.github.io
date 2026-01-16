---
title: "学生练习：Python数据可视化实战"  
collection: portfolio  
type: "Machine Learning"  
permalink: /portfolio/python-data-visualization-practice  
date: 2026-01-16  
excerpt: "通过Matplotlib、Seaborn等工具实现数据可视化，覆盖EDA、模型评估、无监督学习及SHAP解释，提升数据分析与机器学习中的可视化能力。"  
header:  
  teaser: /images/portfolio/python-data-visualization-practice/age_distribution.png  
tags:  
- 数据可视化  
- EDA  
- 机器学习  
- 模型评估  
- SHAP  
tech_stack:  
- name: Python  
- name: Matplotlib  
- name: Seaborn  
- name: Scikit-learn  
- name: SHAP  
- name: Pandas  
- name: Numpy  
---  

## 项目背景  
本项目聚焦Python数据可视化在机器学习全流程中的应用，包括探索性数据分析（EDA）、模型评估、无监督学习聚类及模型解释。通过实战练习，掌握常用可视化工具的使用，提升数据洞察与模型可解释性能力。  

## 核心实现  

### 1. 探索性数据分析（EDA）  
**年龄分布直方图**：  
```python  
plt.figure(figsize=(8,5))  
sns.histplot(data=picu_data, x='age_month', kde=True)  
plt.title("年龄分布直方图")  
plt.show()  
```  

**不同结局下指标箱线图**：  
```python  
colname = ['age_month', 'lab_5237_min', 'lab_5227_min', 'lab_5225_range', 'lab_5235_max', 'lab_5257_min']  
fig, axs = plt.subplots(3,2, constrained_layout=True, figsize=(10,10))  
for i in range(len(colname)):  
    sns.boxplot(data=picu_data, x='HOSPITAL_EXPIRE_FLAG', y=colname[i], ax=axs[i//2, i%2])  
plt.suptitle("不同结局下各实验室指标分布", fontsize=16)  
plt.show()  
```  

### 2. 模型评估可视化  
**混淆矩阵热力图**：  
```python  
confusion_matrix_plot(y_true=y_test, y_pred_prob=y_pred_prob, threshold=0.5)  
```  

**ROC曲线**：  
```python  
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)  
roc_auc = roc_auc_score(y_test, y_pred_prob)  
plt.figure(figsize=(6,5))  
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC={roc_auc:.2f})')  
plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')  
plt.xlabel('假阳性率')  
plt.ylabel('真阳性率')  
plt.title(f'ROC曲线 (AUC={roc_auc:.4f})')  
plt.legend(loc="lower right")  
plt.show()  
```  

### 3. 无监督学习（KMeans聚类）  
**肘部法则图**：  
```python  
wcss = []  
for i in range(1,11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  
    kmeans.fit(data_clustering_scaled)  
    wcss.append(kmeans.inertia_)  
plt.figure(figsize=(10,5))  
plt.plot(range(1,11), wcss, marker='o', linestyle='--')  
plt.title('肘部法则')  
plt.xlabel('簇数')  
plt.ylabel('簇内平方和（WCSS）')  
plt.grid(True)  
plt.show()  
```  

### 4. 模型解释（SHAP）  
**SHAP蜂群图**：  
```python  
shap.summary_plot(shap_values, X_train)  
```  

## 分析结果  
![年龄分布直方图](/images/portfolio/python-data-visualization-practice/age_distribution.png)  
**分析**：年龄分布呈现正态趋势，核密度曲线显示集中在某一区间。  

![ROC曲线](/images/portfolio/python-data-visualization-practice/roc_curve.png)  
**分析**：ROC曲线下面积（AUC）为XX，模型区分能力良好。  

![SHAP蜂群图](/images/portfolio/python-data-visualization-practice/shap_beeswarm.png)  
**分析**：特征lab_5235_max对模型预测影响最大，高值会增加死亡风险。  
