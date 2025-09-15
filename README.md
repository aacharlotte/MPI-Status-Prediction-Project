# MPI-Status-Prediction-Project


This project applies Classical ML and neural networks to predict multidimensional poverty using data from the 2019/2020 Uganda National Household Survey (UNHS) data. The UNHS collect information on socio-economic characteristics at individual, household and community level is used. The dataset provides detailed information on demographics, education, health, living standards and employment, which are used to compute the Multidimensional Poverty Index (MPI). The main objective of the project is to build a predictive model that can classify individuals as multidimansionally poor or non-poor, to enable policymakers design interventions that target individual. 

The project was conducted in several stages: (1) data preprocessing, including handling missing values, scaling numerical variables, and encoding categorical features; (2) model development,using both classical ML models and neural network with Keras and TensorFlow; (3) hyperparameter tuning with GridSearchCV (for classical ML models) Keras Tuner (for neural networks) to identify the best set of hyperparameters and network architecture; (4) training and evaluation with metrics such as accuracy, precision, recall, F1-score, and visualization of learning curves (for nearal networks); and (5) saving best model for deployment. Deploying the best model for future predictions in underway using Streamlit. In this project, we showcase how advanced AI techniques can support evidence-based policymaking for poverty alleviation.

---

### Tools and Libraries Used
- **Machine Learning and Neural Networks (Keras/TensorFlow)** for prediction.
- **GridSearchCV and Keras Tuner** for hyperparameter optimization.
- **Early Stopping & Dropout** to avoid overfitting.
- **ModelCheckpoint** for saving the best-performing models.
- **Preprocessing pipelines (scikit-learn)** for scaling, imputation, and encoding.
- **Visualization (Matplotlib and seaborn)** for EDA, model evaluation, and training/validation performance.
- **Deployment-ready pipeline** with support for **Streamlit** 


---

## Visualizations and Interpretation

### Correlation Heatmap
<img src="https://github.com/aacharlotte/Diabetes-Prediction/blob/main/correlation%20heatmap.png" width="120%" />

**Key Observations:**
- No very high correlations between features were identified. 

---


Beyond household poverty prediction, this project enables:

Scholarship targeting → school-aged poor children.

Nutrition programs → women of reproductive age, children under 5.

Employment programs → unemployed youth.

Elderly care → vulnerable older individuals.







# Diabetes Prediction

For this project, I use the CDC Diabetes Health Indicators dataset to illustrate my approach to model selection. I first do predictions with a baseline model and a couple of untuned models using the train test data. I also balance the dataset (given that there is a class imbalance), do hyperparameter tuning, apply the best thresholds, and use PCA (using different variation retention). The best model is selected depending on the performance metrics ideal for the problem at hand. For this particular project, the best-performing model is determined based on the F1 score since the aim is to ensure correct prediction of diabetic cases to target them for medication. F1 score gives a balace between precision and recall.  


### Tools and Libraries Used
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations  
- **Seaborn**: Data visualization  
- **Matplotlib**: Plotting and customization
- **time**: Track model training time with PCA 

---

## Problem Statement
 Predict whether an individual is diabetic based on health indicators such as BMI, smoking, physical activity, etc.

---

## Aim of the Project
Build a classification model that ensures that both classes (diabetic and non-diabetic) are detected well, with priority given to detecting all diabetic cases to target these for proper diagnosis.

---

## Skills and Concepts Demonstrated
- EDA  
- Preprocessing
- Train-test split
- Building a baseline model  
- Building 6 classification models and improving their performance with class balance, hyperparameter tuning, and threshold tuning
- Interpreting model performance metrics 

---

## Visualizations and Interpretation

### Correlation Heatmap
<img src="https://github.com/aacharlotte/Diabetes-Prediction/blob/main/correlation%20heatmap.png" width="120%" />

**Key Observations:**
- No very high correlations between features were identified. 

---

### Baseline model performance
<img src="https://github.com/aacharlotte/Diabetes-Prediction/blob/main/Baseline%20%20Logistic%20Regression%20ConfusionMatrix.png" width="60%" />

**Key Observations:**
- The baseline model provided performance reference and presented data challenges to be handled with advanced models.
- The main challenge presented was poor performance regarding predicting diabetic cases. The models performed well in predicting non-diabetic cases but predicted the diabetic cases poorly. This is mainly due to class imbalance. The baseline model presented the need for improving model performance with class balance, hyperparameter tuning, and threshold tuning
---

### Best model: Random Forest
<img src="https://github.com/aacharlotte/Diabetes-Prediction/blob/main/Best%20model%20-%20Random%20Forest%20ConfusionMatrix.png" width="60%" />

**Key Observations:**
- Of the six models built, Random Forest emerged as the best-performing model with the highest recall and F1-score (as also seen in the table below). The better recall indicates that this model misses fewer real diabetic cases compared to other models. 

---

### Baseline model performance
<img src="https://github.com/aacharlotte/Diabetes-Prediction/blob/main/Model%20performance.png" width="100%" />


---

## Author & License
**Author**: [Charlotte Arinaitwe]  
**Date**: 2025  
**License**: MIT

---
