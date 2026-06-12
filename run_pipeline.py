import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Ensure directories exist
output_dir = "C:\\Users\\B SRAVANTHI\\.gemini\\antigravity\\scratch\\ai_internship_tasks"
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "customer_data.csv")

# Generate data if not present
if not os.path.exists(csv_path):
    print("Dataset not found. Generating synthetic dataset first...")
    from generate_data import create_synthetic_data
    create_synthetic_data()

print("--- 1. Loading and Cleaning Data ---")
df = pd.read_csv(csv_path)

# Show initial info
print(f"Dataset shape: {df.shape}")
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Basic Cleaning:
# Convert TotalCharges to numeric (just in case), coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median
total_charges_median = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)

print("\nMissing values per column after cleaning:")
print(df.isnull().sum())

print("\n--- 2. Exploratory Data Analysis (EDA) ---")
# Summary statistics
summary_stats = df.describe(include='all')
print("\nSummary Statistics:")
print(summary_stats)

# Plotting
sns.set_theme(style="whitegrid")

# Plot 1: Histogram (Age Distribution)
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], kde=True, color='skyblue', bins=20)
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '1_age_histogram.png'))
plt.close()

# Plot 2: Boxplot (Tenure by Churn Status)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='Tenure', hue='Churn', data=df, palette='Set2', legend=False)
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Churn (0 = Retained, 1 = Churned)')
plt.ylabel('Tenure (Months)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '2_tenure_boxplot.png'))
plt.close()

# Plot 3: Scatter Plot (Monthly Charges vs Total Charges)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df, alpha=0.7, palette='coolwarm')
plt.title('Monthly Charges vs Total Charges')
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Total Charges ($)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '3_charges_scatter.png'))
plt.close()

# Plot 4: Bar Chart (Contract Type Counts)
plt.figure(figsize=(8, 5))
sns.countplot(x='ContractType', hue='ContractType', data=df, palette='viridis', legend=False)
plt.title('Customer Count by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '4_contract_bar.png'))
plt.close()

print(f"4 basic EDA plots saved to: {plot_dir}")

print("\n--- 3. Simple Linear Regression ---")
# Feature: MonthlyCharges, Target: TotalCharges
# Note: For simple linear regression, we predict TotalCharges based on MonthlyCharges
X_reg = df[['MonthlyCharges']]
y_reg = df['TotalCharges']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

print(f"Linear Regression Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Model Coefficient (Slope): {reg_model.coef_[0]:.2f}")
print(f"Model Intercept: {reg_model.intercept_:.2f}")

print("\n--- 4. Supervised Classification (Churn Prediction) ---")
# Preprocessing for classification
# One-hot encode ContractType and Gender
df_encoded = pd.get_dummies(df.drop(columns=['CustomerID']), columns=['ContractType', 'Gender'], drop_first=True)

X_clf = df_encoded.drop(columns=['Churn'])
y_clf = df_encoded['Churn']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Models
lr_model = LogisticRegression(max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
lr_model.fit(X_train_clf, y_train_clf)
rf_model.fit(X_train_clf, y_train_clf)

# Predict
y_pred_lr = lr_model.predict(X_test_clf)
y_prob_lr = lr_model.predict_proba(X_test_clf)[:, 1]

y_pred_rf = rf_model.predict(X_test_clf)
y_prob_rf = rf_model.predict_proba(X_test_clf)[:, 1]

# Cross-Validation scores (Accuracy)
cv_lr = cross_val_score(lr_model, X_clf, y_clf, cv=5, scoring='accuracy').mean()
cv_rf = cross_val_score(rf_model, X_clf, y_clf, cv=5, scoring='accuracy').mean()

# Calculate metrics
metrics = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test_clf, y_pred_lr),
        "Precision": precision_score(y_test_clf, y_pred_lr),
        "Recall": recall_score(y_test_clf, y_pred_lr),
        "F1-Score": f1_score(y_test_clf, y_pred_lr),
        "ROC-AUC": roc_auc_score(y_test_clf, y_prob_lr),
        "5-Fold CV Accuracy": cv_lr
    },
    "Random Forest": {
        "Accuracy": accuracy_score(y_test_clf, y_pred_rf),
        "Precision": precision_score(y_test_clf, y_pred_rf),
        "Recall": recall_score(y_test_clf, y_pred_rf),
        "F1-Score": f1_score(y_test_clf, y_pred_rf),
        "ROC-AUC": roc_auc_score(y_test_clf, y_prob_rf),
        "5-Fold CV Accuracy": cv_rf
    }
}

metrics_df = pd.DataFrame(metrics).T
print("\nClassification Model Comparison:")
print(metrics_df.round(4))

# Plot ROC Curves
plt.figure(figsize=(8, 6))
for name, model, prob in [("Logistic Regression", lr_model, y_prob_lr), ("Random Forest", rf_model, y_prob_rf)]:
    fpr, tpr, _ = roc_curve(y_test_clf, prob)
    auc_val = roc_auc_score(y_test_clf, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '5_roc_curves.png'))
plt.close()

print(f"ROC Curves plot saved to: {plot_dir}")
