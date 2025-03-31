import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# === Helper Function ===
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def eval_model(true, pred, name):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    print(f"{name:<20} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return rmse, mae, r2

def run_pipeline(target, data_dir, save_prefix):
    print(f"\n=== Running for target: {target} ===")

    # === Prepare Sequences ===
    X = scaled_df[features].values
    y = scaled_df[target].values

    time_steps = 24
    X_seq, y_seq = create_sequences(X, y, time_steps=time_steps)
    X_flat = X[time_steps:]
    y_flat = y[time_steps:]

    split = int(0.8 * len(X_seq))
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

    X_train_flat, X_test_flat = X_flat[:split], X_flat[split:]
    y_train_flat, y_test_flat = y_flat[:split], y_flat[split:]

    # === LSTM ===
    lstm_model = Sequential([
        LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    y_pred_lstm = lstm_model.predict(X_test_seq).flatten()

    # === Other Models ===
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, y_train_flat)
    y_pred_rf = rf.predict(X_test_flat)

    svm = SVR()
    svm.fit(X_train_flat, y_train_flat)
    y_pred_svm = svm.predict(X_test_flat)

    lr = LinearRegression()
    lr.fit(X_train_flat, y_train_flat)
    y_pred_lr = lr.predict(X_test_flat)

    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train_flat, y_train_flat)
    y_pred_xgb = xgb.predict(X_test_flat)

    # === Hybrid Meta Model ===
    meta_X = np.vstack([
        y_pred_lstm,
        y_pred_rf,
        y_pred_svm,
        y_pred_lr,
        y_pred_xgb,
        np.mean([y_pred_rf, y_pred_svm, y_pred_lr], axis=0),
        np.std([y_pred_rf, y_pred_svm, y_pred_lr], axis=0),
        np.abs(y_pred_rf - y_test_seq),
    ]).T

    meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4)
    meta_model.fit(meta_X, y_test_seq)
    y_pred_hybrid = meta_model.predict(meta_X)

    # === Evaluation ===
    print(f"\n📊 Evaluation Metrics for {target}:")
    metrics = {}
    metrics['LSTM'] = eval_model(y_test_seq, y_pred_lstm, "LSTM")
    metrics['Random Forest'] = eval_model(y_test_seq, y_pred_rf, "Random Forest")
    metrics['SVM'] = eval_model(y_test_seq, y_pred_svm, "SVM")
    metrics['Linear Regression'] = eval_model(y_test_seq, y_pred_lr, "Linear Regression")
    metrics['XGBoost'] = eval_model(y_test_seq, y_pred_xgb, "XGBoost")
    metrics['Hybrid'] = eval_model(y_test_seq, y_pred_hybrid, "Hybrid Ensemble")

    # === Forecast Plot ===
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_seq[:100], label='Actual', linestyle='--', linewidth=2, color='black')
    plt.plot(y_pred_lstm[:100], label='LSTM')
    plt.plot(y_pred_rf[:100], label='RF')
    plt.plot(y_pred_svm[:100], label='SVM')
    plt.plot(y_pred_lr[:100], label='LR')
    plt.plot(y_pred_xgb[:100], label='XGB')
    plt.plot(y_pred_hybrid[:100], label='Hybrid', linewidth=2.5, linestyle='-.', color='red')
    plt.xlabel("Time Step")
    plt.ylabel(f"{target} (Scaled)")
    plt.title(f"Forecast Comparison: {target} (First 100 Hours)")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"{save_prefix}_forecast_100h.png"), dpi=600)
    plt.show()

    # === Scatter Plot ===
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_seq, y_pred_hybrid, alpha=0.5, edgecolors='k')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Scatter: Hybrid Prediction vs Actual ({target})")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"{save_prefix}_scatter.png"), dpi=600)
    plt.show()

# === Load and Preprocess ===
data_dir = "C:/Users/cavus/Desktop/Conference_SAT-Guard"
power = pd.read_csv(os.path.join(data_dir, "power_D.csv"))
weather = pd.read_csv(os.path.join(data_dir, "weather_for_D.csv"))

power.rename(columns={"Date & Time": "datetime"}, inplace=True)
weather.rename(columns={"time": "datetime"}, inplace=True)
power["datetime"] = pd.to_datetime(power["datetime"], format="%d/%m/%Y %H:%M")
weather["datetime"] = pd.to_datetime(weather["datetime"], format="%d/%m/%Y %H:%M")
data = pd.merge(power, weather, on="datetime")
data.columns = [c.strip().replace("&", "and").replace(";", "").replace(",", "").replace(" ", "_") for c in data.columns]

features = [
    'temperature', 'humidity', 'visibility', 'apparentTemperature', 'pressure',
    'windSpeed', 'cloudCover', 'precipIntensity', 'dewPoint', 'precipProbability'
]

data.fillna(data.median(numeric_only=True), inplace=True)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[features + ['use_[kW]', 'gen_[kW]']])
scaled_df = pd.DataFrame(scaled, columns=features + ['use_[kW]', 'gen_[kW]'])

# === Run both use_[kW] and gen_[kW]
run_pipeline('use_[kW]', data_dir, 'use_kw')
run_pipeline('gen_[kW]', data_dir, 'gen_kw')













