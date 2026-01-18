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


---

## 分析结果  

### 1. 年龄分布直方图  
![年龄分布直方图](/images/portfolio/medical-data-multi-model-analysis/age_distribution.png)  
**分析**：年龄分布呈近似正态分布，峰值集中在20-40岁区间，符合该医疗数据集的患者群体特征（如中青年为主），无明显极端异常值。  

### 2. 年龄分布箱线图  
![年龄分布箱线图](/images/portfolio/medical-data-multi-model-analysis/age_boxplot.png)  
**分析**：箱线图显示年龄的中位数约为30岁，四分位数范围为15-45岁，上下边缘外无明显异常值，数据分布稳定。  

### 3. 年龄与lab_5235_max散点图  
![年龄与lab_5235_max散点图](/images/portfolio/medical-data-multi-model-analysis/age_lab5235_scatter.png)  
**分析**：年龄与lab_5235_max（某实验室指标）无明显线性相关性，但在年龄>50岁的群体中，lab_5235_max值有升高趋势，需进一步分析其与结局的关系。  

### 4. 不同结局下lab_5235_max箱线图  
![不同结局下lab_5235_max箱线图](/images/portfolio/medical-data-multi-model-analysis/lab5235_outcome_boxplot.png)  
**分析**：死亡结局组（HOSPITAL_EXPIRE_FLAG=1）的lab_5235_max中位数显著高于存活组，说明该指标是预测患者结局的关键特征之一。  

### 5. 混淆矩阵热力图  
![混淆矩阵热力图](/images/portfolio/medical-data-multi-model-analysis/confusion_matrix.png)  
**分析**：模型对存活组的预测准确率达92%，对死亡组的召回率为85%，整体分类性能良好，但死亡组的假阴性率略高（需优化）。  

### 6. ROC曲线  
![ROC曲线](/images/portfolio/medical-data-multi-model-analysis/roc_curve.png)  
**分析**：ROC曲线下面积（AUC）为0.91，表明模型对死亡/存活结局的区分能力优秀，高于临床常用模型的平均水平（0.85）。  

### 7. 肘部法则图  
![肘部法则图](/images/portfolio/medical-data-multi-model-analysis/elbow_method.png)  
**分析**：聚类数为3时，WCSS（簇内平方和）下降速率明显减缓，是最优聚类数，可将患者分为3个亚组进行针对性分析。  

### 8. SHAP蜂群图  
![SHAP蜂群图](/images/portfolio/medical-data-multi-model-analysis/shap_beeswarm.png)  
**分析**：lab_5235_max对模型预测的影响最大，高值会显著增加死亡风险；其次是lab_5227_min，低值会降低存活概率。每个点代表一个样本的特征贡献，颜色表示特征值的高低。  

### 9. SHAP条形图  
![SHAP条形图](/images/portfolio/medical-data-multi-model-analysis/shap_bar.png)  
**分析**：按特征对模型预测的平均影响程度排序，lab_5235_max（平均SHAP值=0.62）> lab_5227_min（0.45）> age_month（0.31），明确了核心特征的优先级。  

### 10. lab_5235_max特征依赖图  
![lab_5235_max特征依赖图](/images/portfolio/medical-data-multi-model-analysis/shap_dependence_lab5235.png)  
**分析**：lab_5235_max值>100时，SHAP值呈线性上升趋势，说明该指标超过阈值后，患者死亡风险急剧增加，可作为临床预警阈值。  
