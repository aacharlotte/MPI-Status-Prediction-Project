# MPI-Status-Prediction-Project


This project applies Classical ML and neural networks to predict multidimensional poverty using data from the 2019/2020 Uganda National Household Survey (UNHS) data. The UNHS collect information on socio-economic characteristics at individual, household and community level is used. The dataset provides detailed information on demographics, education, health, living standards and employment, which are used to compute the Multidimensional Poverty Index (MPI). The main objective of the project is to build a predictive model that can classify individuals as multidimansionally poor or non-poor, to enable policymakers design interventions that target individual. 

The project was conducted in several stages: (1) data preprocessing, including handling missing values, scaling numerical variables, and encoding categorical features; (2) model development,using both classical ML models and neural network with Keras and TensorFlow; (3) hyperparameter tuning with GridSearchCV (for classical ML models) Keras Tuner (for neural networks) to identify the best set of hyperparameters and network architecture; (4) training and evaluation with metrics such as accuracy, precision, recall, F1-score, and visualization of learning curves (for nearal networks); and (5) saving best model for deployment. Deploying the best model for future predictions is underway using Streamlit (Checkout link at the end of this readme). 

***In this project, we showcase how advanced AI techniques can support evidence-based policymaking for poverty alleviation.***

---

## Tools and Libraries Used
- **Machine Learning and Neural Networks (Keras/TensorFlow)** for prediction.
- **GridSearchCV and Keras Tuner** for hyperparameter optimization.
- **Early Stopping & Dropout** to avoid overfitting.
- **ModelCheckpoint** for saving the best-performing models.
- **Preprocessing pipelines (scikit-learn)** for scaling, imputation, and encoding.
- **Visualization (Matplotlib and seaborn)** for EDA, model evaluation, and training/validation performance.
- **Deployment-ready pipeline** with support for **Streamlit** 


---

## Visualizations and Interpretation

### EDA Visualizations

### Age by Poverty Status
<img src="https://github.com/aacharlotte/MPI-Status-Prediction-Project/blob/main/Age%20by%20MPI.png" width="50%" />

**Key Observations:**
- Each box shows the distribution of age per MPI status group. Median for poor group lower than that of non-poor group. The interquatile range is narrower for the poor group and concentrated in the young age bracket of 5 to 25, with the upper age limit rarely exceeding 6o years.
- This shows that poverty is more common among children and youth.
- It implies that youth-focused deprivations could be key poverty drivers in this context. As such, variables related to educaton and school attendance are strong MPI status predictors
-  

---


### Employment by Poverty Status
<img src="https://github.com/aacharlotte/MPI-Status-Prediction-Project/blob/main/Employment%20by%20MPI.png" width="50%" />

**Key Observations:**
- Looking at the two categories, there is a higher number of poor people in the unemployed category (~12500) vs 9000 in the employed category. This suggests being employed generally reduces the risk of multidimensional poverty.
- However, the fact that many not employed individuals are also not poor suggests that they may have some other means of support or income.
- In conclusion, employment status is a key factor for predicting poverty, but combining it with other variables improves model accuracy.
  
---
### ML Visualizations

### Logistic Regression
<img src="https://github.com/aacharlotte/MPI-Status-Prediction-Project/blob/main/logistic%20reg.png" width="50%" />

**Key Observations:**
- Best Parameters according to GridSearchCV: {'model__C': 1, 'model__class_weight': 'balanced', 'model__l1_ratio': 0.5, 'model__penalty': 'elasticnet', 'model__solver': 'saga'}
- Accuracy (0.69): Low accuracy. Lower than our baseline model, even.
- Recall (0.73) vs Precision (0.67): This model is better at accurately predicting that a true poor individual is poor (true positives). It is more aggressive at predicting the positive class. This is important in our use case because its imperative that as many poor people as possible are correctly identified and served.
- However, this increased aggression has also increased the number of false positives, negatively affecting our precision score.

***On a general note, our tuned model has higher recall and precision scores than the baseline. This has also led to an improved F1-score. However, accuracy has reduced, likely due to increased false positives from our more aggressive positive-seeking model.***
 

---

### Random Forest
<img src="https://github.com/aacharlotte/MPI-Status-Prediction-Project/blob/main/Random%20Forest.png" width="50%" />

**Key Observations:**
- **Accuracy:** Has high accuracy of 90%
- **Precision vs Recall:** Higher precision for the negative class (0.91) than positive class (0.89) but both generally very good. Model seems slightly more conservative about labeling a prediction positive (poor). Some true positives (truly poor) can be missed due to this.

**Comparison of Averages**
-Macro Avg (0.85): Quite a good score.
Weighted Avg (0.90): Because the negative class has many more samples, this average is pulled much closer to its performance and matches the overall accuracy
 

---

## Comparison of the model performances
<img src="https://github.com/aacharlotte/MPI-Status-Prediction-Project/blob/main/ML%20model%20comparison%20(2).png" width="120%" />

**Ranking comparative model performance**
1. Random forest classifier
2. Decision Tree classifier
3. Tuned logistic regression model
4. Baseline Logistic regression model

**Overall Comparisons:**
- The tuned models have identical, very high macro recall+precision scores (~0.82). However, their accuracy is drastically lower (0.69 and 0.82 respectively) than the Random Forest's (0.90).
- This difference suggests that the tuned LogReg and Decision Tree likely sacrificed performance on the majority class to boost performance on the minority class. This is a common outcome when using techniques like class_weight='balanced'. The Random Forest had a high performance on the minority class without sacrificing performance on the majority class. This clearly makes it the best performing model.
- The Decision tree outcompetes the tuned LogReg model when it comes to overall accuracy.
- The baseline LogReg is the poorest performer of all and highlights the importance of hyperparameter tuning.

---
### NN Visualizations

### NN with selected features
**Training Loss:**

<img src="https://github.com/aacharlotte/MPI-Status-Prediction-Project/blob/main/NN%20loss.jpeg" width="50%" />

---

**NN Model Performance:**

<img src="https://github.com/aacharlotte/MPI-Status-Prediction-Project/blob/main/NN%20classification%20report%20(2).png" width="50%" />

---

# Streamlit App
https://aacharlotte-mpi-status-prediction-app1-0k9nru.streamlit.app/




 


