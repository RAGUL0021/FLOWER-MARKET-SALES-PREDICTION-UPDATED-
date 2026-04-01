import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False


st.title(" Flower Market Demand & Price Prediction System")

# LOAD DATASET
df = pd.read_csv("Flower_market_dataset.csv")

st.subheader(" Full Dataset")
st.dataframe(df)


# SAFE COLUMN HANDLING
def ensure_column(col, default="No"):
    if col not in df.columns:
        df[col] = default

ensure_column("Festival")
ensure_column("Wedding")
ensure_column("Holiday")

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

df["Season"] = df["Month"].apply(get_season)

valid_seasons = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]


# ENCODING
le_flower = LabelEncoder()
le_season = LabelEncoder()
le_supplier = LabelEncoder()

df["Flower_Encoded"] = le_flower.fit_transform(df["Flower_Type"])
df["Season_Encoded"] = le_season.fit_transform(df["Season"])
df["Supplier_Encoded"] = le_supplier.fit_transform(df["Supplier"])


# VISUALIZATION

# PIE CHART
st.subheader(" Season-wise Demand")

season_demand = (
    df.groupby("Season")["Quantity_Sold_kg"]
    .sum()
    .reindex(valid_seasons)
)

fig1, ax1 = plt.subplots()
ax1.pie(season_demand, labels=season_demand.index, autopct="%1.1f%%")
ax1.set_title("Season-wise Demand")
st.pyplot(fig1)


# BAR CHART
st.subheader(" Season-wise Flower-wise Demand")

season_flower_demand = (
    df.groupby(["Season", "Flower_Type"])["Quantity_Sold_kg"]
    .sum()
    .unstack()
    .fillna(0)
)

fig2 = season_flower_demand.T.plot(kind="bar", figsize=(12,6)).figure
st.pyplot(fig2)


# FEATURES
features = [
    "Flower_Encoded",
    "Season_Encoded",
    "Supplier_Encoded",
    "Wholesale_Price_per_kg",
    "Competitor_Price",
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

st.subheader(" Model Performance")

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Convert regression → classification for accuracy_score
    tolerance = 0.1 * y_test
    y_true_class = (y_test > y_test.mean()).astype(int)
    y_pred_class = (preds > y_test.mean()).astype(int)

    acc = accuracy_score(y_true_class, y_pred_class)

    results.append([name, mae, mse, r2, acc])
    trained_models[name] = model

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "MSE", "R2 Score", "Accuracy"]
)

st.dataframe(results_df)

best_model_name = results_df.sort_values("R2 Score", ascending=False).iloc[0]["Model"]
best_demand_model = trained_models[best_model_name]

st.success(f"Best Model: {best_model_name}")


# PRICE MODEL
price_model = RandomForestRegressor(random_state=42)
price_model.fit(X, y_price)


# PRICE GRAPH
st.subheader(" Season-wise Flower Price Prediction")

df["Predicted_Price"] = price_model.predict(X)

season_flower_price = (
    df.groupby(["Season", "Flower_Type"])["Predicted_Price"]
    .mean()
    .unstack()
    .fillna(0)
)

fig3 = season_flower_price.T.plot(kind="bar", figsize=(12,6)).figure
st.pyplot(fig3)


# LIVE PREDICTION
st.subheader(" Live Prediction")

flower_input = st.selectbox("Flower Type", le_flower.classes_)
season_input = st.selectbox("Season", valid_seasons)
supplier_input = st.selectbox("Supplier", le_supplier.classes_)

wholesale_price = st.number_input("Wholesale Price ₹")
competitor_price = st.number_input("Competitor Price ₹")

festival = st.selectbox("Festival", ["Yes", "No"])
wedding = st.selectbox("Wedding", ["Yes", "No"])
holiday = st.selectbox("Holiday", ["Yes", "No"])
weekend = st.selectbox("Weekend", ["Yes", "No"])

month = st.slider("Month", 1, 12)

if st.button("Predict"):

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

    pred_demand = best_demand_model.predict(input_df)[0]
    pred_price = price_model.predict(input_df)[0]

    st.success(f" Predicted Demand: {pred_demand:.2f} kg")
    st.success(f" Predicted Price: ₹{pred_price:.2f}")


# NEXT DAY PREDICTION
from datetime import datetime, timedelta

st.subheader(" Next Day Prediction")

if st.button("Predict Next Day"):

    next_day = datetime.today() + timedelta(days=1)
    next_month = next_day.month
    next_season = get_season(next_month)
    next_weekend = 1 if next_day.weekday() >= 5 else 0

    next_input = pd.DataFrame([[
        le_flower.transform([flower_input])[0],
        le_season.transform([next_season])[0],
        le_supplier.transform([supplier_input])[0],
        wholesale_price,
        competitor_price,
        0, 0, 0,
        next_weekend,
        next_month
    ]], columns=features)

    next_demand = best_demand_model.predict(next_input)[0]
    next_price = price_model.predict(next_input)[0]

    st.info(f"Date: {next_day.strftime('%Y-%m-%d')}")
    st.info(f"Season: {next_season}")
    st.success(f" Next Day Demand: {next_demand:.2f} kg")
    st.success(f" Next Day Price: ₹{next_price:.2f}")
