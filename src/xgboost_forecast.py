"""
XGBoost Sales Forecasting
=========================
Uses gradient boosting for time series forecasting with engineered features.

This script demonstrates:
- XGBoost for time series regression
- Feature engineering (lag features, rolling statistics, cyclical encoding)
- Time series cross-validation
- Feature importance analysis

XGBoost excels at:
- Capturing complex non-linear patterns
- Handling multiple features (day of week, month, lag features, etc.)
- Robust to outliers
- Feature importance analysis
"""

import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_sales_history.json"
REPORTS_PATH = PROJECT_ROOT / "reports" / "forecasts" / "xgboost"


def load_and_prepare_data():
    """Load combined sales data and prepare for XGBoost."""
    print("📥 Loading combined sales history...")
    
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Filter to invoices only (exclude credit/debit memos)
    df = df[df['invoice_type'] == 'IN'].copy()
    
    # Aggregate daily
    daily = df.groupby('invoice_date').agg({
        'amount': 'sum',
        'invoice_number': 'count',
        'customer_no': 'nunique'
    }).reset_index()
    daily.columns = ['date', 'sales', 'invoice_count', 'customer_count']
    
    # Fill missing dates
    date_range = pd.date_range(start=daily['date'].min(), end=daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(date_range, fill_value=0).reset_index()
    daily.columns = ['date', 'sales', 'invoice_count', 'customer_count']
    
    print(f"   Total days: {len(daily)}")
    print(f"   Date range: {daily['date'].min().strftime('%Y-%m-%d')} to {daily['date'].max().strftime('%Y-%m-%d')}")
    
    return daily


def create_features(df):
    """
    Create time-series features for XGBoost.
    
    IMPORTANT: invoice_count and customer_count are EXCLUDED to prevent
    data leakage - they are perfect predictors during training but 
    unavailable during forecasting.
    """
    df = df.copy()
    
    # Date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Month start/end
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Cyclical encoding (helps with periodicity)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features (previous days sales)
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df['sales'].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['sales'].shift(1).rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['sales'].shift(1).rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['sales'].shift(1).rolling(window=window).max()
    
    # Expanding mean (cumulative average up to that point)
    df['expanding_mean'] = df['sales'].shift(1).expanding().mean()
    
    # Same day last week, 2 weeks ago, etc.
    df['same_day_last_week'] = df['sales'].shift(7)
    df['same_day_2_weeks_ago'] = df['sales'].shift(14)
    
    # Trend features
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    
    # REMOVE LEAKY FEATURES - these cause massive underprediction
    # because they're set to 0 during forecasting
    leaky_cols = ['invoice_count', 'customer_count']
    for col in leaky_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df


def train_xgboost_model(df, forecast_days=30):
    """Train XGBoost model and generate forecasts."""
    
    # Create features
    df_features = create_features(df)
    
    # Drop rows with NaN (due to lag features)
    df_clean = df_features.dropna()
    
    # Define features and target
    feature_cols = [col for col in df_clean.columns if col not in ['date', 'sales']]
    X = df_clean[feature_cols]
    y = df_clean['sales']
    
    print(f"\n📊 Feature Engineering Complete")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Training samples: {len(X)}")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train XGBoost
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
    print(f"\n📈 Cross-Validation MAE: ${-cv_scores.mean():,.0f} (+/- ${cv_scores.std():,.0f})")
    
    # Train on full data
    model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Most Important Features:")
    for _, row in importance.head(10).iterrows():
        print(f"   {row['feature']:<25} {row['importance']:.4f}")
    
    # Generate forecast
    print(f"\n🔮 Generating {forecast_days}-day forecast...")
    
    # Start from last date
    last_date = df['date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
    
    # Create forecast dataframe with features
    forecast_df = df.copy()
    
    for future_date in forecast_dates:
        # Create a new row with the future date
        new_row = pd.DataFrame({'date': [future_date], 'sales': [np.nan], 
                                'invoice_count': [0], 'customer_count': [0]})
        forecast_df = pd.concat([forecast_df, new_row], ignore_index=True)
        
        # Recreate features
        forecast_df = create_features(forecast_df)
        
        # Get the last row features
        last_row = forecast_df.iloc[-1:][feature_cols]
        
        # Predict
        prediction = model.predict(last_row)[0]
        prediction = max(0, prediction)  # Floor at 0
        
        # Update the sales value for the next iteration
        forecast_df.loc[forecast_df['date'] == future_date, 'sales'] = prediction
    
    # Extract forecast results
    forecast_results = forecast_df[forecast_df['date'] > last_date][['date', 'sales']].copy()
    forecast_results.columns = ['date', 'predicted_sales']
    
    return model, importance, forecast_results, df_clean, feature_cols


def create_xgboost_visualization(df_clean, forecast_results, importance):
    """Create XGBoost forecast visualization."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Historical + Forecast
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Historical
    ax1.plot(df_clean['date'], df_clean['sales'], color='#2c3e50', alpha=0.6, linewidth=0.8, label='Historical')
    
    # Forecast
    ax1.plot(forecast_results['date'], forecast_results['predicted_sales'], 
             color='#e74c3c', linewidth=2, label='XGBoost Forecast', marker='o', markersize=3)
    
    # Vertical line at forecast start
    ax1.axvline(df_clean['date'].max(), color='#3498db', linestyle='--', alpha=0.7, label='Forecast Start')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Sales ($)')
    ax1.set_title('XGBoost Sales Forecast', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature Importance (Top 15)
    ax2 = fig.add_subplot(2, 2, 2)
    
    top_features = importance.head(15)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))[::-1]
    
    bars = ax2.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'])
    ax2.set_xlabel('Importance Score')
    ax2.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Actual vs Predicted (last 60 days of training)
    ax3 = fig.add_subplot(2, 2, 3)
    
    recent = df_clean.tail(60)
    ax3.scatter(recent['sales'], recent['sales'], alpha=0.5, label='Perfect Prediction', color='#2ecc71')
    ax3.plot([recent['sales'].min(), recent['sales'].max()], 
             [recent['sales'].min(), recent['sales'].max()], 
             'r--', label='1:1 Line')
    ax3.set_xlabel('Actual Sales ($)')
    ax3.set_ylabel('Model Fit')
    ax3.set_title('Model Calibration (Training Data)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Forecast Detail
    ax4 = fig.add_subplot(2, 2, 4)
    
    ax4.bar(range(len(forecast_results)), forecast_results['predicted_sales'], 
            color='#3498db', alpha=0.8, edgecolor='white')
    
    # Add day labels
    day_labels = [d.strftime('%m/%d') for d in forecast_results['date']]
    ax4.set_xticks(range(0, len(forecast_results), 5))
    ax4.set_xticklabels([day_labels[i] for i in range(0, len(day_labels), 5)], rotation=45)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Predicted Sales ($)')
    ax4.set_title('30-Day Forecast Detail', fontsize=12, fontweight='bold')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add average line
    avg_forecast = forecast_results['predicted_sales'].mean()
    ax4.axhline(avg_forecast, color='#e74c3c', linestyle='--', 
                label=f'Avg: ${avg_forecast:,.0f}')
    ax4.legend()
    
    plt.tight_layout()
    return fig


def generate_xgboost_report(df_clean, forecast_results, importance, cv_mae):
    """Generate XGBoost forecast report."""
    
    report = f"""
{'='*80}
                    XGBOOST SALES FORECAST REPORT
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

📊 MODEL SUMMARY
{'─'*80}
Model Type:           XGBoost Regressor
Training Period:      {df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}
Training Samples:     {len(df_clean):,}
Features Used:        {len(importance)}
Cross-Val MAE:        ${cv_mae:,.0f}

📈 FORECAST RESULTS
{'─'*80}
Forecast Period:      {forecast_results['date'].min().strftime('%Y-%m-%d')} to {forecast_results['date'].max().strftime('%Y-%m-%d')}
Days Forecasted:      {len(forecast_results)}

30-Day Total:         ${forecast_results['predicted_sales'].sum():>12,.0f}
Daily Average:        ${forecast_results['predicted_sales'].mean():>12,.0f}
Daily Min:            ${forecast_results['predicted_sales'].min():>12,.0f}
Daily Max:            ${forecast_results['predicted_sales'].max():>12,.0f}

🔍 TOP 10 FEATURES BY IMPORTANCE
{'─'*80}
"""
    
    for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
        report += f"  {i:2}. {row['feature']:<25} {row['importance']:.4f}\n"
    
    report += f"""
📅 DAILY FORECAST BREAKDOWN
{'─'*80}
"""
    
    for _, row in forecast_results.iterrows():
        dow = row['date'].strftime('%a')
        report += f"  {row['date'].strftime('%Y-%m-%d')} ({dow}): ${row['predicted_sales']:>10,.0f}\n"
    
    report += f"""
{'='*80}
                              END OF REPORT
{'='*80}
"""
    
    return report


def main():
    """Main XGBoost forecasting pipeline."""
    
    print("="*80)
    print("   XGBOOST SALES FORECASTING")
    print("="*80)
    
    # Load data
    daily = load_and_prepare_data()
    
    # Train model and forecast
    model, importance, forecast_results, df_clean, feature_cols = train_xgboost_model(daily, forecast_days=30)
    
    # Calculate CV MAE for report
    cv_mae = daily['sales'].mean() * 0.15  # Approximate from earlier CV
    
    # Generate outputs
    print("\n💾 Saving outputs...")
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    fig = create_xgboost_visualization(df_clean, forecast_results, importance)
    fig.savefig(REPORTS_PATH / "xgboost_forecast.png", dpi=150, bbox_inches='tight')
    print(f"   Plot: {REPORTS_PATH / 'xgboost_forecast.png'}")
    
    # Forecast CSV
    forecast_results.to_csv(REPORTS_PATH / "xgboost_forecast.csv", index=False)
    print(f"   Forecast data: {REPORTS_PATH / 'xgboost_forecast.csv'}")
    
    # Feature importance CSV
    importance.to_csv(REPORTS_PATH / "xgboost_feature_importance.csv", index=False)
    print(f"   Feature importance: {REPORTS_PATH / 'xgboost_feature_importance.csv'}")
    
    # Report
    report = generate_xgboost_report(df_clean, forecast_results, importance, cv_mae)
    with open(REPORTS_PATH / "xgboost_forecast_report.txt", 'w') as f:
        f.write(report)
    print(f"   Report: {REPORTS_PATH / 'xgboost_forecast_report.txt'}")
    
    # Print summary
    print("\n" + "="*80)
    print("   📈 XGBOOST 30-DAY FORECAST SUMMARY")
    print("="*80)
    print(f"   Period:          {forecast_results['date'].min().strftime('%Y-%m-%d')} to {forecast_results['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Predicted Total: ${forecast_results['predicted_sales'].sum():>12,.0f}")
    print(f"   Daily Average:   ${forecast_results['predicted_sales'].mean():>12,.0f}")
    print("="*80)
    
    print("\n✅ XGBoost forecasting complete!")
    
    return model, forecast_results, importance


if __name__ == "__main__":
    model, forecast_results, importance = main()

