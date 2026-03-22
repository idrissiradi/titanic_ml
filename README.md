# Titanic Survival Prediction — Feature Engineering & Classification
 
Binary classification project predicting passenger survival on the Titanic. Focuses on **feature engineering from raw text fields**, proper `Pipeline` construction to prevent data leakage, and model evaluation with a real Kaggle submission.
 
---

## Results
 
| Metric | Score |
|---|---|
| Local test accuracy | **81%** |
| Kaggle public leaderboard | **77%** |
 
---
## Pipeline
 
```
train.csv / test.csv
       │
       ▼
Feature Engineering
  · Title extracted from Name  (Mr / Mrs / Miss / Master / Rare)
  · FamilySize = SibSp + Parch + 1
  · IsAlone flag
  · CabinDeck from first char of Cabin
  · AgeBand (binned)
       │
       ▼
sklearn Pipeline
  · SimpleImputer   → Age (median), Embarked (most_frequent)
  · OrdinalEncoder  → categorical features
  · StandardScaler  → numeric features
       │
       ▼
LogisticRegression
  · 81% local accuracy
  · 77% Kaggle score
       │
       ▼
submission.csv  →  Kaggle
```
 
---
 
## Stack
 
- **Python 3.x**, Jupyter Notebook
- **scikit-learn** — LogisticRegression, Pipeline, SimpleImputer, StandardScaler, OrdinalEncoder
- **pandas / NumPy** — feature engineering
- **Matplotlib / Seaborn** — EDA visualizations
 
---
 
## Setup
 
```bash
git clone https://github.com/idrissiradi/titanic_ml
cd titanic_ml
pip install -r requirements.txt
jupyter notebook main.ipynb
```
 
---
