# Sales ML Forecasting Demo

A comprehensive sales forecasting project demonstrating multiple machine learning approaches for time series prediction.

> **Note:** This project uses synthetic sample data that preserves realistic statistical patterns (seasonality, trends) while protecting confidential business information. The code and techniques are production-ready examples.

## Overview

This project implements and compares four forecasting approaches:

1. **Prophet** - Facebook's time series forecasting tool
2. **SARIMAX** - Seasonal ARIMA with exogenous variables
3. **XGBoost** - Gradient boosting for time series regression
4. **Scikit-Learn Ensemble** - Multiple ML models with stacking

Plus:
- **Ensemble Forecasting** - Weighted combination of all models
- **Model Validation** - Proper time series backtesting
- **Churn Prediction** - Customer risk scoring

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone this repository
git clone https://github.com/java-heapler/sales-forecasting-demo.git
cd sales-forecasting-demo

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
sales-ml-demo/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   └── processed/
│       └── combined_sales_history.json  # Sample sales data
├── src/
│   ├── generate_sample_data.py   # Creates synthetic data
│   ├── prophet_tuned.py          # Prophet with hyperparameter tuning
│   ├── sarimax_forecast.py       # SARIMAX forecasting
│   ├── xgboost_forecast.py       # XGBoost forecasting
│   ├── sklearn_forecast.py       # Multiple sklearn models
│   ├── ensemble_forecast.py      # Combines all models
│   ├── model_validation.py       # Backtesting framework
│   └── churn_sklearn.py          # Customer churn prediction
├── reports/                      # Generated outputs
│   ├── forecasts/
│   │   ├── prophet/
│   │   ├── sarimax/
│   │   ├── xgboost/
│   │   ├── sklearn/
│   │   └── ensemble/
│   ├── validation/
│   └── customers/churn/
└── models/                       # Saved model files
```

## Quick Start

### Option A: Run Everything at Once
```bash
python run_all.py
```
This runs all models in sequence and generates all reports.

### Option B: Run Individual Scripts

#### 1. Generate Sample Data (or use included data)

```bash
python src/generate_sample_data.py
```

This creates realistic synthetic sales data with:
- ~1000 days of daily sales records
- 500 fake customers
- Weekly seasonality (high Mon-Fri, low weekends)
- Monthly and yearly trends
- Realistic variance

### 2. Run Individual Forecasting Models

```bash
# Prophet (with hyperparameter tuning)
python src/prophet_tuned.py

# SARIMAX
python src/sarimax_forecast.py

# XGBoost
python src/xgboost_forecast.py

# Scikit-learn (multiple models)
python src/sklearn_forecast.py
```

### 3. Run Ensemble Forecast

```bash
python src/ensemble_forecast.py
```

### 4. Validate Model Performance

```bash
python src/model_validation.py
```

### 5. Run Churn Prediction

```bash
python src/churn_sklearn.py
```

## Learning Topics

### Time Series Forecasting

| Model | Best For | Key Concepts |
|-------|----------|--------------|
| **Prophet** | Seasonality, holidays | Decomposable models, changepoints, Fourier seasonality |
| **SARIMAX** | Statistical rigor, confidence intervals | Stationarity, autocorrelation, differencing |
| **XGBoost** | Complex patterns, feature importance | Gradient boosting, lag features, rolling statistics |
| **Sklearn** | Model comparison, ensembles | Stacking, voting, cross-validation |

### Key Concepts Demonstrated

1. **Feature Engineering for Time Series**
   - Lag features (yesterday's sales, last week's sales)
   - Rolling statistics (7-day mean, 14-day std)
   - Cyclical encoding (sin/cos for day of week)
   - Calendar features (month end, quarter, holidays)

2. **Time Series Cross-Validation**
   - Why random splits don't work for time series
   - TimeSeriesSplit for proper validation
   - Walk-forward validation

3. **Ensemble Methods**
   - Weighted averaging based on validation performance
   - Stacking with meta-learner
   - Uncertainty from model disagreement

4. **Avoiding Data Leakage**
   - Features that shouldn't be used (invoice count, customer count)
   - Proper train/test splits
   - Point-in-time feature generation

5. **Hyperparameter Tuning**
   - Grid search for Prophet parameters
   - Impact of regularization on forecasts
   - Balancing bias vs variance

### Churn Prediction Topics

- RFM features (Recency, Frequency, Monetary)
- Customer lifecycle modeling
- Classification metrics (AUC, precision, recall)
- Feature importance for interpretability

## Data Schema

The `combined_sales_history.json` file contains records with:

```json
{
  "invoice_number": "INV-1000042",
  "invoice_date": "2024-03-15",
  "customer_no": "CUST0042",
  "account_number": "00-CUST0042",
  "customer_name": "Alpine Water Distribution",
  "amount": 1234.56,
  "invoice_type": "IN",
  "source": "Sample"
}
```

## Expected Output

After running all scripts, you'll have:

### Forecasts
- 30-day predictions from each model
- Ensemble forecast combining all models
- Confidence intervals

### Visualizations
- Model comparison charts
- Feature importance plots
- Seasonal pattern analysis
- Cumulative forecast projections

### Reports
- Model performance metrics
- Validation results
- Daily forecast breakdowns

## Common Issues

### "Prophet not found"
```bash
pip install prophet
```
Note: Prophet may require additional dependencies. See [Prophet Installation](https://facebook.github.io/prophet/docs/installation.html).

### "XGBoost not found"
```bash
pip install xgboost
```

### Slow model training
- Reduce `n_estimators` in XGBoost/sklearn models
- Reduce parameter grid size in Prophet tuning
- Use fewer CV folds

## Contributing

This is a demonstration project. Feel free to:
- Experiment with different models
- Add new features
- Adjust hyperparameters
- Try different ensemble weights

## License

MIT License - Feel free to use for learning and projects.

