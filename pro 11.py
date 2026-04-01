
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False


# LOAD YOUR DATASET

df = pd.read_csv("Flower_market_dataset.csv")


# SAFE COLUMN HANDLING

def ensure_column(col, default="No"):
    if col not in df.columns:
        df[col] = default

ensure_column("Festival")
ensure_column("Wedding")
ensure_column("Holiday")

# Weekend handling 
if "Weekend" not in df.columns:
    df["Weekend"] = df["Weekend_Days"]

for col in ["Festival", "Wedding", "Holiday", "Weekend"]:
    df[col] = df[col].astype(str).map({"Yes": 1, "No": 0}).fillna(0)


# DATE & SEASON 

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Month"] = df["Date"].dt.month

def get_season(month):
    if month in [3, 4]:
        return "Spring"
    elif month in [5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    elif month in [10, 11]:
        return "Autumn"
    else:
        return "Winter"

# Override existing Season column
df["Season"] = df["Month"].apply(get_season)

valid_seasons = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]


# ENCODING

le_flower = LabelEncoder()
le_season = LabelEncoder()
le_supplier = LabelEncoder()

df["Flower_Encoded"] = le_flower.fit_transform(df["Flower_Type"])
df["Season_Encoded"] = le_season.fit_transform(df["Season"])
df["Supplier_Encoded"] = le_supplier.fit_transform(df["Supplier"])


# VISUALIZATIONS

#  Season-wise Demand
season_demand = (
    df.groupby("Season")["Quantity_Sold_kg"]
    .sum()
    .reindex(valid_seasons)
)

plt.figure()
plt.pie(season_demand, labels=season_demand.index, autopct="%1.1f%%")
plt.title("Season-wise Demand")
plt.show()


#  Season-wise Flower-wise Demand

season_flower_demand = (
    df.groupby(["Season", "Flower_Type"])["Quantity_Sold_kg"]
    .sum()
    .unstack()
    .fillna(0)
)


season_flower_demand.T.plot(kind="bar", figsize=(12,6))

plt.title("Season-wise Flower-wise Demand")
plt.xlabel("Flower Type")
plt.ylabel("Quantity (kg)")
plt.legend(title="Season")

plt.show()


# FEATURES & TARGET 

features = [
    "Flower_Encoded",
    "Season_Encoded",
    "Supplier_Encoded",
    "Wholesale_Price_per_kg",
    "Competitor_Price",   # your dataset column
    "Festival",
    "Wedding",
    "Holiday",
    "Weekend",
    "Month"
]

X = df[features]
y_demand = df["Quantity_Sold_kg"]
y_price = df["Retail_Price_per_kg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_demand, test_size=0.2, random_state=42
)


# MODELS 

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

print("\nModel Results:")

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append([name, mae, r2])
    trained_models[name] = model

    print(f"{name} → MAE: {mae:.2f}, R2: {r2:.2f}")

results_df = pd.DataFrame(results, columns=["Model", "MAE", "R2 Score"])

best_model_name = results_df.sort_values("R2 Score", ascending=False).iloc[0]["Model"]
best_demand_model = trained_models[best_model_name]

print(f"\nBest Demand Model: {best_model_name}")


# PRICE MODEL

price_model = RandomForestRegressor(random_state=42)
price_model.fit(X, y_price)


#  SEASON-WISE FLOWER-WISE PRICE PREDICTION GRAPH


df["Predicted_Price"] = price_model.predict(X)

season_flower_price = (
    df.groupby(["Season", "Flower_Type"])["Predicted_Price"]
    .mean()
    .unstack()
    .fillna(0)
)

# NO plt.figure() here
season_flower_price.T.plot(kind="bar", figsize=(12,6))

plt.title("Season-wise Flower-wise Price Prediction")
plt.xlabel("Flower Type")
plt.ylabel("Predicted Price ₹")
plt.legend(title="Season")

plt.show()



# LIVE PREDICTION (PYTHON VERSION - USER INPUT)

print("\n LIVE PREDICTION")

# User Inputs
flower_input = input(f"Enter Flower Type {list(le_flower.classes_)}: ")
season_input = input(f"Enter Season {valid_seasons}: ")
supplier_input = input(f"Enter Supplier {list(le_supplier.classes_)}: ")

wholesale_price = float(input("Enter Wholesale Price ₹: "))
competitor_price = float(input("Enter Competitor Price ₹: "))

festival = input("Festival (Yes/No): ")
wedding = input("Wedding (Yes/No): ")
holiday = input("Holiday (Yes/No): ")
weekend = input("Weekend (Yes/No): ")

month = int(input("Enter Month (1-12): "))

# Convert inputs
input_df = pd.DataFrame([[
    le_flower.transform([flower_input])[0],
    le_season.transform([season_input])[0],
    le_supplier.transform([supplier_input])[0],
    wholesale_price,
    competitor_price,
    1 if festival == "Yes" else 0,
    1 if wedding == "Yes" else 0,
    1 if holiday == "Yes" else 0,
    1 if weekend == "Yes" else 0,
    month
]], columns=features)

# Predictions
pred_demand = best_demand_model.predict(input_df)[0]
pred_price = price_model.predict(input_df)[0]

print(f"\n Predicted Demand: {pred_demand:.2f} kg")
print(f" Predicted Retail Price: ₹{pred_price:.2f}")


# NEXT DAY PREDICTION (AUTO LOGIC)

print("\n NEXT DAY PREDICTION")

from datetime import datetime, timedelta

# Get next day
today = datetime.today()
next_day = today + timedelta(days=1)

next_month = next_day.month

# Season from next day
def get_season(month):
    if month in [3, 4]:
        return "Spring"
    elif month in [5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    elif month in [10, 11]:
        return "Autumn"
    else:
        return "Winter"

next_season = get_season(next_month)

# Weekend check
next_weekend = 1 if next_day.weekday() >= 5 else 0

# Default assumptions (can improve later)
festival_flag = 0
wedding_flag = 0
holiday_flag = 0

# Use same flower & supplier from user
next_input = pd.DataFrame([[
    le_flower.transform([flower_input])[0],
    le_season.transform([next_season])[0],
    le_supplier.transform([supplier_input])[0],
    wholesale_price,        # assume same price
    competitor_price,       # assume same
    festival_flag,
    wedding_flag,
    holiday_flag,
    next_weekend,
    next_month
]], columns=features)

# Next day predictions
next_demand = best_demand_model.predict(next_input)[0]
next_price = price_model.predict(next_input)[0]

print(f"\n Date: {next_day.strftime('%Y-%m-%d')}")
print(f"Season: {next_season}")
print(f" Next Day Demand: {next_demand:.2f} kg")
print(f" Next Day Price: ₹{next_price:.2f}")