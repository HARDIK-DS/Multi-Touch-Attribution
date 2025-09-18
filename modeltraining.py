import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from featureengineering import add_features
from preprocessing import encode_columns, load_data

# Load and preprocess
df = load_data('data/final_shop_6modata.csv')
df = add_features(df, is_training=True)   # ✅ Now adds KPIs
df, le_adgroup, le_month = encode_columns(df)

# Features & target
features = [
    'Impressions', 'Clicks', 'CTR', 'Conversions', 'Conv Rate',
    'Cost', 'CPC', 'Sale Amount', 'P&L',
    'ROI', 'Profit_Margin', 'CPM', 'Revenue_per_Click', 'Revenue_per_Conversion',
    'Ad_Group_Encoded', 'Month_Encoded'
]

target = 'Revenue'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    objective='reg:squarederror'
)
xgb_model.fit(X_train, y_train)

# Evaluation
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Save model and encoders
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(le_adgroup, "le_adgroup.pkl")
joblib.dump(le_month, "le_month.pkl")
print("Model and encoders saved successfully!")
