import pandas as pd
import numpy as np
import os

def create_synthetic_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    
    # 1. Customer IDs
    customer_ids = [f"{i:04d}-XYZ" for i in range(1, num_samples + 1)]
    
    # 2. Demographics & Account Info
    age = np.random.randint(18, 80, size=num_samples)
    gender = np.random.choice(["Male", "Female"], size=num_samples)
    
    # Contract Type
    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract_probs = [0.5, 0.25, 0.25]
    contract = np.random.choice(contract_choices, size=num_samples, p=contract_probs)
    
    # Tenure (correlated with contract type)
    tenure = []
    for c in contract:
        if c == "Month-to-month":
            tenure.append(np.random.randint(1, 24))
        elif c == "One year":
            tenure.append(np.random.randint(12, 48))
        else:
            tenure.append(np.random.randint(24, 72))
    tenure = np.array(tenure)
    
    # Monthly Charges
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=num_samples), 2)
    
    # Total Charges (with some noise and linear relationship)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 50, size=num_samples), 2)
    # Ensure no negative total charges
    total_charges = np.clip(total_charges, a_min=20.0, a_max=None)
    
    # Introduce missing values in TotalCharges (approx 3%) for cleaning practice
    missing_idx = np.random.choice(num_samples, size=int(num_samples * 0.03), replace=False)
    total_charges_raw = total_charges.astype(object)
    total_charges_raw[missing_idx] = np.nan
    
    # 3. Target Variable: Churn (influenced by tenure, monthly charges, and contract type)
    # Map contract types to numerical risk factor using np.select
    contract_risk = np.select(
        [contract == "Month-to-month", contract == "One year"],
        [1.0, 0.2],
        default=0.05
    )
    
    # Probability of churn calculation
    churn_prob = (
        0.4 * (1.0 - tenure / 72.0) +  # Short tenure increases churn
        0.3 * (monthly_charges / 120.0) +  # High monthly charges increase churn
        0.3 * contract_risk
    )
    # Add random noise and clip probability
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.1, size=num_samples), 0, 1)
    
    # Convert probability to binary label (0 or 1)
    churn = (np.random.rand(num_samples) < churn_prob).astype(int)
    
    # Combine into DataFrame
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_raw,
        "ContractType": contract,
        "Churn": churn
    })
    
    # Save to CSV
    os.makedirs("C:\\Users\\B SRAVANTHI\\.gemini\\antigravity\\scratch\\ai_internship_tasks", exist_ok=True)
    csv_path = "C:\\Users\\B SRAVANTHI\\.gemini\\antigravity\\scratch\\ai_internship_tasks\\customer_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated successfully at: {csv_path}")
    print(df.head())

if __name__ == "__main__":
    create_synthetic_data()
