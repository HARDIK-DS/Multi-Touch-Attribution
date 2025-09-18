import pandas as pd
import joblib
from featureengineering import add_features

# Load trained model & encoders
xgb_model = joblib.load("xgb_model.pkl")
le_adgroup = joblib.load("le_adgroup.pkl")
le_month = joblib.load("le_month.pkl")

# Required features for prediction
features = [
    'Impressions', 'Clicks', 'CTR', 'Conversions', 'Conv Rate',
    'Cost', 'CPC', 'Sale Amount', 'P&L',
    'ROI', 'Profit_Margin', 'CPM', 'Revenue_per_Click', 'Revenue_per_Conversion',
    'Ad_Group_Encoded', 'Month_Encoded'
]

# ------------------ Helper functions ------------------

def encode_input(df):
    # Encode 'Ad Group'
    if 'Ad Group' in df.columns:
        df['Ad_Group_Encoded'] = df['Ad Group'].apply(
            lambda x: le_adgroup.transform([x])[0] if x in le_adgroup.classes_ else 0
        )
    else:
        df['Ad_Group_Encoded'] = 0

    # Encode 'Month'
    if 'Month' in df.columns:
        df['Month_Encoded'] = df['Month'].apply(
            lambda x: le_month.transform([x])[0] if x in le_month.classes_ else 0
        )
    else:
        df['Month_Encoded'] = 0

    return df

def add_kpis(df, revenue_col='Predicted_Revenue'):
    # Compute KPIs using predicted or actual revenue
    df['ROI'] = df[revenue_col] / (df['Cost'] + 1e-6)
    df['Profit_Margin'] = df['P&L'] / (df[revenue_col] + 1e-6)
    df['CPM'] = df['Cost'] / (df['Impressions'] + 1e-6) * 1000
    df['Revenue_per_Click'] = df[revenue_col] / (df['Clicks'] + 1e-6)
    df['Revenue_per_Conversion'] = df[revenue_col] / (df['Conversions'] + 1e-6)
    return df

# ------------------ Prediction functions ------------------

def predict_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = add_features(df, is_training=False)
    df = encode_input(df)

    # Fill missing features with 0
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X_input = df[features]
    df['Predicted_Revenue'] = xgb_model.predict(X_input)

    # Compute KPIs using predicted revenue
    df = add_kpis(df, revenue_col='Predicted_Revenue')

    # Display results
    cols_to_show = ['Ad Group', 'Month', 'Predicted_Revenue', 'ROI', 'Profit_Margin',
                    'CPM', 'Revenue_per_Click', 'Revenue_per_Conversion']
    print(df[cols_to_show])

    # Save predictions
    df.to_csv("predicted_revenue_with_kpis.csv", index=False)
    print("Predictions saved as predicted_revenue_with_kpis.csv")

def predict_manual():
    print("Enter campaign details manually:")
    data = {}

    numeric_cols = ['Impressions', 'Clicks', 'CTR', 'Conversions',
                    'Cost', 'CPC', 'Sale Amount', 'P&L']
    for col in numeric_cols:
        val = input(f"{col}: ")
        data[col] = [float(val) if val else 0]

    data['Ad Group'] = [input("Ad Group: ") or "Unknown"]
    data['Month'] = [input("Month: ") or "Unknown"]

    df = pd.DataFrame(data)
    df = add_features(df, is_training=False)
    df = encode_input(df)

    # Fill missing features
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X_input = df[features]
    df['Predicted_Revenue'] = xgb_model.predict(X_input)

    # Compute KPIs
    df = add_kpis(df, revenue_col='Predicted_Revenue')

    # Display results
    cols_to_show = ['Ad Group', 'Month', 'Predicted_Revenue', 'ROI', 'Profit_Margin',
                    'CPM', 'Revenue_per_Click', 'Revenue_per_Conversion']
    print("\nPrediction with KPIs:")
    print(df[cols_to_show])

# ------------------ Main ------------------

if __name__ == "__main__":
    mode = input("Select mode: 1 = CSV upload, 2 = Manual input: ")

    if mode.strip() == "1":
        csv_path = input("Enter path to CSV file: ")
        predict_csv(csv_path)
    elif mode.strip() == "2":
        predict_manual()
    else:
        print("Invalid mode selected. Please choose 1 or 2.")
