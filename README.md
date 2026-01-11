# Uber Surge Pricing Prediction (Enhanced)

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
