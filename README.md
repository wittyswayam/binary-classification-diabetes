# 🩺 Pima Indians Diabetes Prediction

A machine learning project to predict the onset of diabetes in patients using the Pima Indians Diabetes Dataset. This project walks through the full data science pipeline — from exploratory data analysis (EDA) to model training, evaluation, and deployment-ready predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Project Walkthrough](#project-walkthrough)
  - [Step 1: Data Loading](#step-1-data-loading)
  - [Step 2: Exploratory Data Analysis (EDA)](#step-2-exploratory-data-analysis-eda)
  - [Step 3: Train-Test Split](#step-3-train-test-split)
  - [Step 4: Model Training](#step-4-model-training)
    - [Logistic Regression](#41-logistic-regression)
    - [Random Forest](#42-random-forest)
  - [Step 5: Deployment / Prediction](#step-5-deployment--prediction)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🔍 Overview

Diabetes is a chronic health condition affecting millions worldwide. Early and accurate prediction of diabetes risk can help clinicians intervene proactively. This project builds binary classification models to predict whether a patient will develop diabetes (`Outcome = 1`) or not (`Outcome = 0`) based on medical diagnostic measurements.

The project demonstrates:
- How to handle real-world tabular medical data
- EDA techniques including distribution analysis and correlation heatmaps
- Training and evaluating classical ML classifiers
- Saving trained models with `pickle` for reuse
- Making live predictions from a trained model

---

## 📊 Dataset

**Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

| Property | Value |
|---|---|
| Total Samples | 768 |
| Features | 8 |
| Target Variable | `Outcome` (0 = No Diabetes, 1 = Diabetes) |
| Class Distribution | ~65% Non-Diabetic, ~35% Diabetic |

### Feature Description

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (µU/ml) |
| `BMI` | Body mass index (weight in kg / height in m²) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (genetic likelihood score) |
| `Age` | Age in years |
| `Outcome` | Target: 1 = diabetic, 0 = non-diabetic |

> ⚠️ **Note on Zero Values:** Several features (e.g., Glucose, BloodPressure, BMI, Insulin) contain zero values that are biologically impossible and likely represent missing data. The EDA step investigates this.

---

## 📁 Project Structure

```
binary-classification-diabetes/
│
├── diabetes.csv                          # Raw dataset (768 rows × 9 columns)
├── Pima_Diabetes_Final_Project.ipynb     # Main Jupyter Notebook (full pipeline)
└── README.md                             # Project documentation (this file)
```

**Generated outputs** (after running the notebook):
```
├── logistic_regression_model.pkl         # Saved Logistic Regression model
└── random_forest_model.pkl               # Saved Random Forest model
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab (recommended)

### Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Clone the Repository

```bash
git clone https://github.com/wittyswayam/binary-classification-diabetes.git
cd Diabetes-projects
```

### Launch Jupyter Notebook

```bash
jupyter notebook "Pima_Diabetes_Final_Project.ipynb"
```

Or open directly in **Google Colab** by uploading the `.ipynb` file.

> 📌 If running on Colab, update the file path in the data loading cell:
> ```python
> df = pd.read_csv("/content/diabetes.csv")
> ```

---

## 🗺️ Project Walkthrough

### Step 1: Data Loading

```python
import numpy as np
import pandas as pd

df = pd.read_csv("/content/diabetes.csv")
print("Data loaded. Shape:", df.shape)
df.head(10)
```

The dataset is loaded into a pandas DataFrame. With `df.shape` returning `(768, 9)`, we confirm 768 patient records across 9 columns (8 features + 1 target).

---

### Step 2: Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic info and statistics
df.info()
df.describe()

# Check for zero values (proxy for missing data)
zero_counts = (df == 0).sum()

# Outcome distribution
sns.countplot(x='Outcome', data=df)

# Feature histograms
df.hist(bins=20, figsize=(12, 10))

# Correlation heatmap
sns.heatmap(df.corr(), annot=True)
```

**Key EDA findings:**
- **Class imbalance:** ~500 non-diabetic vs ~268 diabetic samples
- **Zero values** in `Insulin` (374 zeros), `SkinThickness` (227 zeros), `BloodPressure` (35 zeros), `Glucose` (5 zeros), and `BMI` (11 zeros) — these are likely missing values
- **Glucose** is the most strongly correlated feature with the target (`Outcome`)
- **BMI** and **Age** also show notable positive correlations with diabetes

---

### Step 3: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

| Split | Samples |
|---|---|
| Training Set (80%) | ~614 samples |
| Test Set (20%) | ~154 samples |

- **`random_state=42`** ensures reproducible splits
- **`stratify=y`** maintains the class ratio in both train and test sets — important for imbalanced datasets

---

### Step 4: Model Training

#### 4.1 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model
with open("logistic_regression_model.pkl", "wb") as file:
    pickle.dump(model, file)
```

**Why Logistic Regression?**
- A fast, interpretable baseline model
- Well-suited for binary classification problems
- Outputs class probabilities, useful for calibration
- `max_iter=200` is set to ensure convergence on this dataset

---

#### 4.2 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
import pickle

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(model, file)
```

**Why Random Forest?**
- Ensemble method that combines 200 decision trees for robust predictions
- Handles non-linear feature interactions naturally
- Resistant to overfitting compared to a single decision tree
- Can compute feature importances to understand which variables matter most
- `n_estimators=200` provides a good balance between accuracy and training speed

---

### Step 5: Deployment / Prediction

After training, the model can make predictions on new individual patient records:

```python
# Predict a single patient (row index 10 from the dataset)
sample = df.iloc[10, :-1].values.reshape(1, -1)
result = model.predict(sample)

if result[0] == 1:
    print("Diabetes detected")
else:
    print("No diabetes detected")
```

The trained models are serialized to `.pkl` files using Python's `pickle` module. This allows the models to be:
- Loaded and reused without retraining
- Integrated into a web app (e.g., Flask or FastAPI)
- Deployed to cloud platforms for real-time inference

---

## 📈 Results

Both models were evaluated on the held-out test set using classification report and confusion matrix metrics.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | ~77–79% | ~78–80% |
| Precision (Diabetic) | ~72% | ~74% |
| Recall (Diabetic) | ~60–65% | ~62–68% |
| F1-Score (Diabetic) | ~65–68% | ~67–70% |

> 📌 Exact numbers may vary slightly between runs. Random Forest generally outperforms Logistic Regression on this dataset, but Logistic Regression provides better interpretability.

**Confusion Matrix Interpretation:**
- **True Positives (TP):** Diabetic patients correctly identified
- **True Negatives (TN):** Healthy patients correctly identified
- **False Positives (FP):** Healthy patients incorrectly flagged (less costly)
- **False Negatives (FN):** Diabetic patients missed (higher clinical cost)

> ⚠️ In a medical context, **recall** for the positive class (diabetes) is often more important than raw accuracy, as missing a diabetic patient (false negative) has higher consequences.

---

## 🛠️ Technologies Used

| Tool / Library | Purpose |
|---|---|
| **Python 3** | Core programming language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical data visualization |
| **Scikit-learn** | ML models, train-test split, evaluation metrics |
| **Pickle** | Model serialization and saving |
| **Jupyter Notebook** | Interactive development environment |
| **Google Colab** | Cloud-based notebook execution |

---

## ▶️ How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/wittyswayam/binary-classification-diabetes.git
   cd Diabetes-projects
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Open the notebook:**
   ```bash
   jupyter notebook "Pima_Diabetes_Final_Project.ipynb"
   ```

4. **Run all cells** from top to bottom (Kernel → Restart & Run All)

5. **Models will be saved** as `.pkl` files in the current directory after training completes.

---

## 🚀 Future Improvements

- [ ] **Handle missing/zero values** — Impute biologically impossible zeros using median or KNN imputation
- [ ] **Feature scaling** — Apply `StandardScaler` or `MinMaxScaler` for algorithms sensitive to feature magnitude
- [ ] **Hyperparameter tuning** — Use `GridSearchCV` or `RandomizedSearchCV` for optimal model parameters
- [ ] **Additional models** — Experiment with XGBoost, SVM, and Neural Networks

---

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes) / [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Original dataset collected by the National Institute of Diabetes and Digestive and Kidney Diseases

---
