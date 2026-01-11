import os

# --- Content for the Improved Machine Learning Model (main.py) ---
main_code = r'''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def main():
    print("--- Uber Surge Pricing Prediction (Improved) ---")

    # 1. Load Data
    try:
        df = pd.read_csv("uber_surge_data.csv")
        print(">> Data loaded successfully.")
    except FileNotFoundError:
        print("!! Error: 'uber_surge_data.csv' not found.")
        print("   Please make sure the CSV file is in this folder.")
        return

    # 2. Data Cleaning
    # Remove 'ride_id' because it's just a random ID, not a predictive feature.
    if 'ride_id' in df.columns:
        df = df.drop(columns=['ride_id'])
        print(">> Dropped irrelevant 'ride_id' column.")

    # Separate features and target
    X = df.drop(columns=['surge_multiplier_class'])
    y = df['surge_multiplier_class']

    # 3. Preprocessing Setup
    # Encode categorical columns properly so the model understands them
    categorical_cols = ['time_of_day', 'weather', 'location_type', 'demand_level']

    # Use OneHotEncoder (better than LabelEncoder for non-ordered categories)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # 4. Build the Pipeline
    # Preprocess -> Balance Classes (SMOTE) -> Train Model (Random Forest)
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train
    print(">> Training Random Forest model... (this may take a moment)")
    model_pipeline.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print(f"Model Accuracy: {acc*100:.2f}%")
    print("="*30)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Cross Validation (Verification)
    print(">> Running Cross-Validation (checking robustness)...")
    cv_scores = cross_val_score(model_pipeline, X, y, cv=5)
    print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}%")

    # 8. Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Actual vs Predicted')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
'''

# --- Content for the Documentation (README.md) ---
readme_code = r"""# Uber Surge Pricing Prediction (Enhanced)

## Project Overview
This project predicts whether Uber surge pricing will be **Low**, **Medium**, or **High**.

## Why this version is better
1. **Algorithm Upgrade**: Uses **Random Forest** instead of Logistic Regression for better accuracy.
2. **Correct Encoding**: Uses **One-Hot Encoding** so the model doesn't think "Snowy" > "Clear".
3. **Imbalance Handling**: Uses **SMOTE** to generate synthetic data for rare classes.
4. **Noise Reduction**: Removed the `ride_id` column which confuses the model.

## How to Run
1. Ensure `uber_surge_data.csv` is in this folder.
2. Install dependencies:
   `pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn`
3. Run the model:
   `python main.py`
"""

# --- Write the files to the current folder ---
def install_project():
    print("Installing project files...")
    
    with open("main.py", "w") as f:
        f.write(main_code)
        print("✅ Created 'main.py' (The improved AI code)")

    with open("README.md", "w") as f:
        f.write(readme_code)
        print("✅ Created 'README.md' (Documentation)")
        
    print("\nInstallation Complete!")
    print("⚠️  IMPORTANT: Don't forget to copy 'uber_surge_data.csv' into this folder!")
    print("Then run: python main.py")

if __name__ == "__main__":
    install_project()
    