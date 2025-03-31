# NeuroFusion: Weather-Aware Energy Forecasting Framework

**NeuroFusion** is a hybrid deep learning and machine learning framework developed to improve the accuracy and resilience of energy consumption and generation forecasts in smart grid environments. By combining Long Short-Term Memory (LSTM) networks with eXtreme Gradient Boosting (XGBoost) in a meta-learning ensemble structure, NeuroFusion captures both temporal dependencies and non-linear weather-related interactions.

## 🔍 Project Overview

With the growing reliance on renewable energy and the expansion of smart cities, reliable energy forecasting has become critical. NeuroFusion addresses this need by integrating:
- LSTM for long-range temporal pattern learning
- XGBoost for capturing non-linear weather-to-energy relationships
- A Gradient Boosting Regressor for final prediction, using engineered meta-features

This model has been evaluated on real-world weather and power datasets, achieving high predictive accuracy:
- 96.3% R² for energy generation
- 91.3% R² for energy consumption
- **RMSE reduction** of over 74–77% compared to standalone models

## ⚙️ Features

- Hybrid model combining sequential memory and boosted decision trees
- Meta-learning for ensemble prediction using engineered features (e.g., prediction variance)
- Support for both energy generation and consumption forecasting
- Adaptable to seasonal changes and variable weather conditions

## 📁 Repository Structure (example)

```
├── data/
│   └── raw/              # Raw weather and power data
│   └── processed/        # Cleaned and aligned data
├── models/
│   ├── lstm_model.py     # LSTM model definition
│   ├── xgboost_model.py  # XGBoost model training
│   └── ensemble_model.py # Meta-learning ensemble
├── notebooks/
│   └── exploratory.ipynb # Data analysis and visualisation
├── utils/
│   └── preprocessing.py  # Data transformation utilities
├── results/
│   └── plots/            # Performance graphs and comparisons
├── README.md
└── requirements.txt
```

## 📊 Dataset

The project uses the Home D dataset from the Smart* project by UMass Amherst, which includes timestamped meteorological and circuit-level energy data from a residential home.

- Dataset Source: [UMass Smart* Repository](https://traces.cs.umass.edu/docs/traces/smartstar/)
- Features include: temperature, humidity, wind speed, pressure, apparent temperature, etc.

## 🧠 Model Pipeline

1. **Data Preparation**: Merge weather and power data, apply z-score normalisation.
2. **Sequence Creation**: Generate time-windowed sequences for LSTM and flat features for other regressors.
3. **Base Model Training**: Train LSTM, Random Forest, SVM, and Linear Regression.
4. **Meta-Feature Engineering**: Calculate mean, standard deviation, and residuals from base predictions.
5. **Final Ensemble**: Train Gradient Boosting Regressor using stacked meta-features.

## 📈 Performance

| Model          | Energy Consumption R² | Energy Generation R² |
|----------------|------------------------|------------------------|
| LSTM           | 0.0608                 | 0.2753                 |
| Random Forest  | 0.1055                 | 0.2801                 |
| SVM            | -0.2753                | 0.1677                 |
| XGBoost        | 0.3665                 | 0.1729                 |
| **NeuroFusion**| **0.9130**             | **0.9634**             |

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

Recommended Python version: `>=3.8`

## 🧪 How to Run

```bash
python run_neurofusion.py
```

Make sure to configure the paths in the script to point to your data directories.

## 📄 Citation

If you use NeuroFusion in your research, please cite:

> Cavus, M., Jiang, J., & Sun, H. (2025). *Weather-Aware Energy Forecasting with NeuroFusion: A Hybrid Deep Learning and Gradient Boosting Framework*. IEEE SmartGrid Conference.

## 🧠 Future Work

- Integration with EV charging forecasts
- Real-time deployment for grid balancing
- Multi-step ahead prediction capability
