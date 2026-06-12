# AI Internship Tasks Summary Report

This report summarizes the implementation, analysis, and modeling results of the AI internship tasks on a simulated customer churn dataset.

---

## 1. Objective & Setup

The objective of this project is to build practical skills in data preprocessing, exploratory data analysis (EDA), regression, and classification.

- **Environment**: Python virtual environment (`.venv`), with libraries `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
- **Dataset**: A custom-generated dataset (`customer_data.csv`) of 1,000 customers containing attributes: `CustomerID`, `Age`, `Gender`, `Tenure`, `MonthlyCharges`, `TotalCharges` (containing 3% missing values), `ContractType`, and `Churn`.

---

## 2. Preprocessing & Data Cleaning

Data preprocessing steps executed:
1. **Type Conversion**: Evaluated `TotalCharges` to verify numerical type.
2. **Missing Value Imputation**: Identified missing values in `TotalCharges` (approx. 3%) and successfully imputed them using the **median** of the column to prevent data distortion or dropping valuable records.

---

## 3. Exploratory Data Analysis (EDA)

Four primary plots were generated to understand variables and relationships:
- **Histogram**: Visualized the distribution of customer ages, confirming a balanced distribution between 18 and 80 years.
- **Boxplot**: Compared `Tenure` against `Churn`. This showed that customers who churned had a significantly shorter tenure (mostly under 10 months) compared to retained customers.
- **Scatter Plot**: Illustrated `MonthlyCharges` vs. `TotalCharges`. A clear positive correlation is visible, with churn occurrences higher at higher monthly charge values.
- **Bar Chart**: Displayed customer counts across contract types. Month-to-month contracts make up approximately 50% of the customer base, representing the highest risk category.

---

## 4. Simple Linear Regression

A Simple Linear Regression model was implemented to predict `TotalCharges` (dependent variable) using `MonthlyCharges` (independent variable).

- **Mean Absolute Error (MAE)**: Measures average absolute prediction error.
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors.

*Key Findings:*
- Regression line slope and intercept reflect a strong linear relationship matching customer lifetimes.
- The errors (MAE & RMSE) represent the baseline variation introduced by customer tenure lengths.

---

## 5. Classification Model Comparison

We built and compared two algorithms to predict binary customer churn (`Churn` = 0 or 1): **Logistic Regression** and **Random Forest Classifier**.

### Metrics Comparison Table

| Metric | Logistic Regression | Random Forest Classifier |
| :--- | :---: | :---: |
| **Accuracy** | 76.50% | 73.00% |
| **Precision** | 79.45% | 78.83% |
| **Recall** | 87.22% | 81.20% |
| **F1-Score** | 83.15% | 80.00% |
| **ROC-AUC** | 0.8033 | 0.7481 |
| **5-Fold CV Accuracy** | 70.70% | 69.00% |

### Model Selection Insights
- **Logistic Regression** outperformed Random Forest Classifier across all key metrics in this dataset. It achieved a higher Accuracy (76.5% vs 73.0%), F1-Score (83.15% vs 80.00%), and ROC-AUC (0.8033 vs 0.7481).
- **Recall** is critical for churn prediction since missing a churn risk (false negative) is more costly than a false positive. Logistic Regression achieved a higher recall (87.22% vs 81.20%), making it the preferred model for this run.
- The 5-Fold Cross-Validation confirms the stable performance of the Logistic Regression model over Random Forest (70.7% vs 69.0%).


---

## 6. Recommendations & Next Steps

1. **Model Deployment**: Deploy the Random Forest model for real-time churn risk prediction.
2. **Feature Engineering**: Incorporate interaction features (e.g., ratio of monthly charges to total income, tenure-to-age ratio) to further boost accuracy.
3. **Hyperparameter Tuning**: Run Grid Search / Random Search to optimize the number of estimators and max depth for Random Forest.
