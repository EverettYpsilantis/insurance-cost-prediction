# insurance-cost-prediction
Regression pipeline predicting patient insurance charges using Linear Regression and Random Forest, Kaggle hackathon submission

# Patient Insurance Cost Prediction

A regression modeling pipeline predicting individual medical insurance charges 
from patient demographics and health factors. Built as part of the 
PML_M3_Practice_Hackathon_2026, achieving a final Kaggle score of 21,520,797, a 45% improvement over the baseline Linear Regression submission.

## Business Context

Medical insurance pricing must balance affordability for individuals with 
financial sustainability for insurers. This project uses a real-world dataset 
to model what drives insurance charges and predict them as accurately as possible.

## Dataset

6 features, train/test split provided via Kaggle:

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Gender (male/female) |
| bmi | Body mass index |
| children | Number of dependents |
| smoker | Smoking status (yes/no) |
| region | Geographic region (NW/NE/SE/SW) |
| charges | Annual insurance cost [target] |

## Modeling Pipeline

### 1. Preprocessing
Label-encoded categorical variables (sex, smoker, region) into numeric form. 
Preserved a copy of the original dataframe to avoid overwriting raw data.

### 2. Linear Regression — Baseline
80/20 train/validation split. Baseline MSE: **~33,635,210**.
Linear regression struggled to capture nonlinear interactions — particularly 
between BMI and smoking status, which drove high error.

### 3. Random Forest Regressor — Final Model
300 trees, unlimited depth, all cores. Validation MSE: **~20,778,721**, a ~38% reduction over baseline. Random Forest captured the nonlinear 
feature interactions that linear regression missed.

### 4. Learning Curve Analysis
Generated a 5-fold cross-validated learning curve on the Random Forest. 
Revealed mild overfitting (training MSE well below validation MSE) but 
confirmed validation error declined steadily with more data, justifying 
retraining on the full training set before final submission.

### 5. Final Submission
Retrained Random Forest on full training data, predicted on held-out test set, 
submitted to Kaggle. Final score: **21,520,797** (ranked 11th on leaderboard).

## Results

| Model | Validation MSE | Kaggle Score |
|-------|---------------|--------------|
| Linear Regression | 33,635,210 | 39,026,423 |
| Random Forest (300 trees) | 20,778,721 | 21,520,797 |

## Key Takeaways
- Smoking status is the dominant cost driver, smokers carry dramatically 
  higher charges regardless of other factors
- BMI × smoking interactions are nonlinear and require ensemble methods 
  to capture effectively
- Learning curve analysis is a practical diagnostic for overfitting before 
  committing to a final submission

## Tech Stack
Python, scikit-learn, pandas, numpy, matplotlib

## Presentation
See `Predicting_Patient_Insurance_Costs.pdf` for the full project walkthrough.
