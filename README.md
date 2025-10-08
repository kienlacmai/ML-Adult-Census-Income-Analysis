# ML-Adult-Census-Income-Analysis
**Predicting whether an individual's annual income exceeds $50K using UCI’s Adult Census dataset**

---

## Overview
This project applies supervised machine learning techniques to the **Adult Census Income** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu).  
Our goal is to predict whether an individual's annual income exceeds **$50,000 USD** based on demographic and employment features such as **age, education, occupation, and hours worked per week**.

The project includes:
- Exploratory data analysis and preprocessing  
- Training and tuning of multiple classification models  
- Performance comparison and feature importance analysis  

---

## Contributors
- **Kienlac Mai**  
- **Gary Zeng**  
- **Jacky Cheng**

---

## Dataset
- **Source:** [UCI Machine Learning Repository – Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)  
- **Instances:** 48,842  
- **Features:** 14 (both categorical and continuous)  
- **Target Variable:** `income` (≤50K or >50K)

The dataset represents 1994 U.S. Census data. It contains mild class imbalance — about **76% ≤50K** and **24% >50K**, which we addressed through careful train/test splitting while preserving label distribution:contentReference[oaicite:0]{index=0}.

---

## Initial Analysis
Exploratory visualizations revealed key relationships:
- **Age:** Individuals aged **40–50** were most likely to earn >$50K.  
- **Education:** Higher education strongly correlated with income above $50K.  
- **Occupation:** Private sector and professional roles showed higher earning likelihood.  
- **Country & Race:** The majority of data comes from the U.S., with race and nationality distributions contributing to observed biases in income prediction:contentReference[oaicite:1]{index=1}.

---

## Models Implemented
We trained and compared several machine learning classifiers using **80% training / 20% testing splits**, tuning hyperparameters to balance bias and variance.

| Model | Key Parameters | Training Error | Test Error | Accuracy |
|-------|----------------|----------------|-------------|-----------|
| Logistic Regression | Regularization C = {0.1, 1, 10} | 14.89% | 14.62% | 85.38% |
| Gaussian Naive Bayes | var_smoothing ∈ [1e-9, 1] | 20.65% | 19.62% | 80.38% |
| Multinomial Naive Bayes | α ∈ [0.001, 10] | 21.94% | 20.99% | 79.01% |
| Bernoulli Naive Bayes | α ∈ [0.001, 10] | 26.60% | 27.48% | 72.52% |
| Decision Tree | max_depth=7 | 14.17% | 14.62% | 85.38% |
| Random Forest | max_depth=6, n_estimators=46 | 10.46% | 14.16% | 85.84% |

---

## Key Insights
- **Random Forest** achieved the lowest test error and best generalization, confirming its robustness against overfitting.  
- **Logistic Regression** provided interpretable results with similar performance, highlighting influential features such as **marital status, capital gain, and native country**.  
- **Gaussian Naive Bayes** performed well but slightly under the ensemble methods due to mixed data types.  
- Data bias from demographic imbalance (age, race, nationality) likely influenced feature weights and model fairness:contentReference[oaicite:2]{index=2}.

---

## Technologies Used
- **Python**  
- **Pandas**, **NumPy** — data manipulation  
- **Matplotlib**, **Seaborn** — visualization  
- **scikit-learn** — model training and evaluation  
- **Jupyter Notebook** — exploratory workflow  

---

## Results Summary
The **Random Forest Classifier** achieved the best test performance with an error rate of **14.16%**, closely followed by **Logistic Regression (14.62%)**.  
Both models demonstrated strong predictive power and generalization on unseen data:contentReference[oaicite:3]{index=3}.

---
