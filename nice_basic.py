# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 13:57:48 2025

@author: cavus
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# === Load Data ===
data_dir = "C:/Users/cavus/Desktop/Conference_SAT-Guard"
power = pd.read_csv(os.path.join(data_dir, "power_D.csv"))
weather = pd.read_csv(os.path.join(data_dir, "weather_for_D.csv"))

# === Preprocess Data ===
power.rename(columns={"Date & Time": "datetime"}, inplace=True)
weather.rename(columns={"time": "datetime"}, inplace=True)
power["datetime"] = pd.to_datetime(power["datetime"], format="%d/%m/%Y %H:%M")
weather["datetime"] = pd.to_datetime(weather["datetime"], format="%d/%m/%Y %H:%M")
data = pd.merge(power, weather, on="datetime")
data.columns = [c.strip().replace("&", "and").replace(";", "").replace(",", "").replace(" ", "_") for c in data.columns]

# === Feature & Target Columns ===
features = [
    'temperature', 'humidity', 'visibility', 'apparentTemperature', 'pressure',
    'windSpeed', 'cloudCover', 'precipIntensity', 'dewPoint', 'precipProbability'
]
targets = ['use_[kW]', 'gen_[kW]']

# === Handle Missing Values ===
data.fillna(data.median(numeric_only=True), inplace=True)

# === Scale Data ===
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[features + targets])
scaled_df = pd.DataFrame(scaled, columns=features + targets)
scaled_df['datetime'] = data['datetime']

# === Correlation Heatmap ===
plt.figure(figsize=(10, 8))
corr = scaled_df[features + targets].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("🔍 Correlation Heatmap of Features & Energy Metrics")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "correlation_heatmap.png"), dpi=600)
plt.show()

# === Seasonal Variation Plot ===
scaled_df['month'] = scaled_df['datetime'].dt.month
monthly_avg = scaled_df.groupby('month')[[targets[0], targets[1]]].mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg.index, monthly_avg[targets[0]], marker='o', label='Energy Consumption (scaled)')
plt.plot(monthly_avg.index, monthly_avg[targets[1]], marker='s', label='Energy Generation (scaled)')
plt.xticks(range(1, 13))
plt.xlabel("Month", fontsize=16)
plt.ylabel("Energy (scaled) [kW]", fontsize=16)
plt.title("Seasonal Variation in Energy Consumption & Generation")
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "seasonal_variation.png"), dpi=600)
plt.show()

# === Feature Importance with Random Forest ===
rf_use = RandomForestRegressor(n_estimators=100, random_state=42)
rf_use.fit(scaled_df[features], scaled_df[targets[0]])
importance_use = rf_use.feature_importances_

rf_gen = RandomForestRegressor(n_estimators=100, random_state=42)
rf_gen.fit(scaled_df[features], scaled_df[targets[1]])
importance_gen = rf_gen.feature_importances_

# === Feature Importance Plot ===
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].barh(features, importance_use)
axs[0].set_title("🌡️ Feature Importance for Energy Consumption")
axs[0].set_xlabel("Importance")


axs[1].barh(features, importance_gen)
axs[1].set_title("⚡ Feature Importance for Energy Generation")
axs[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig(os.path.join(data_dir, "feature_importance.png"), dpi=600)
plt.show()


# === Colorful Feature Importance Plot ===
import seaborn as sns

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
colors_use = sns.color_palette("viridis", len(features))
colors_gen = sns.color_palette("rocket", len(features))

axs[0].barh(features, importance_use, color=colors_use)
axs[0].set_title("🌡️ Feature Importance for Energy Consumption")
axs[0].set_xlabel("Importance")

axs[1].barh(features, importance_gen, color=colors_gen)
axs[1].set_title("⚡ Feature Importance for Energy Generation")
axs[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig(os.path.join(data_dir, "feature_importance_colorful.png"), dpi=600)
plt.show()


import seaborn as sns

# === Feature Importance for Energy Consumption ===
colors = sns.color_palette("viridis", len(features))

plt.figure(figsize=(8, 6))
plt.barh(features, importance_use, color=colors)
plt.title("🌡️ Feature Importance for Energy Consumption", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(data_dir, "feature_importance_consumption.png"), dpi=600)
plt.show()

# === Feature Importance for Energy Generation ===
plt.figure(figsize=(8, 6))
plt.barh(features, importance_gen, color=colors)
plt.title("⚡ Feature Importance for Energy Generation", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(data_dir, "feature_importance_generation.png"), dpi=600)
plt.show()




# === Triangle Correlation Heatmap ===
plt.figure(figsize=(10, 8))
corr = scaled_df[features + targets].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Triangle heatmap with mask
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            linewidths=0.5, cbar_kws={"shrink": .75})

plt.title("🔍 Correlation Heatmap of Features & Energy Metrics (Lower Triangle)")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "correlation_heatmap_triangle.png"), dpi=600)
plt.show()

