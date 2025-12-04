"""
SARIMAX Sales Forecasting
=========================
Seasonal ARIMA with eXogenous variables for time series forecasting.

This script demonstrates:
- SARIMAX model (Seasonal ARIMA with eXogenous variables)
- Time series stationarity analysis
- Creating exogenous features (day of week, month, etc.)
- Confidence interval forecasting

SARIMAX excels at:
- Capturing seasonal patterns (weekly, monthly, yearly)
- Modeling trend and autocorrelation
- Incorporating external factors (holidays, promotions, etc.)
- Providing confidence intervals
"""

import json
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_sales_history.json"
REPORTS_PATH = PROJECT_ROOT / "reports" / "forecasts" / "sarimax"


def load_and_prepare_data():
    """Load combined sales data and prepare for SARIMAX."""
    print("📥 Loading combined sales history...")
    
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Filter to invoices only
    df = df[df['invoice_type'] == 'IN'].copy()
    
    # Aggregate daily
    daily = df.groupby('invoice_date').agg({
        'amount': 'sum',
        'invoice_number': 'count',
        'customer_no': 'nunique'
    }).reset_index()
    daily.columns = ['date', 'sales', 'invoice_count', 'customer_count']
    
    # Set date as index
    daily = daily.set_index('date')
    
    # Fill missing dates
    daily = daily.asfreq('D', fill_value=0)
    
    print(f"   Total days: {len(daily)}")
    print(f"   Date range: {daily.index.min().strftime('%Y-%m-%d')} to {daily.index.max().strftime('%Y-%m-%d')}")
    
    return daily


def analyze_stationarity(series, title="Sales"):
    """Analyze time series stationarity."""
    print(f"\n📊 Stationarity Analysis for {title}")
    print("─"*60)
    
    # ADF test
    result = adfuller(series.dropna())
    print(f"   ADF Statistic: {result[0]:.4f}")
    print(f"   p-value: {result[1]:.4f}")
    print(f"   Critical Values:")
    for key, value in result[4].items():
        print(f"      {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print("   ✅ Series is stationary (p < 0.05)")
        return True
    else:
        print("   ⚠️ Series is NOT stationary (p >= 0.05)")
        return False


def create_exogenous_features(df):
    """Create exogenous variables for SARIMAX."""
    exog = pd.DataFrame(index=df.index)
    
    # Day of week (one-hot encoded, drop Sunday as reference)
    for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']):
        exog[day] = (df.index.dayofweek == i).astype(int)
    
    # Is weekend
    exog['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Month dummies (drop December as reference)
    for i in range(1, 12):
        exog[f'month_{i}'] = (df.index.month == i).astype(int)
    
    # Quarter
    exog['Q1'] = (df.index.quarter == 1).astype(int)
    exog['Q2'] = (df.index.quarter == 2).astype(int)
    exog['Q3'] = (df.index.quarter == 3).astype(int)
    
    # Month start/end
    exog['month_start'] = df.index.is_month_start.astype(int)
    exog['month_end'] = df.index.is_month_end.astype(int)
    
    # Trend
    exog['trend'] = np.arange(len(df))
    
    return exog


def fit_sarimax_model(daily, exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """Fit SARIMAX model."""
    print(f"\n🔧 Fitting SARIMAX model...")
    print(f"   Order: {order}")
    print(f"   Seasonal Order: {seasonal_order}")
    
    model = SARIMAX(
        daily['sales'],
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=False, maxiter=500)
    
    print(f"   AIC: {results.aic:.2f}")
    print(f"   BIC: {results.bic:.2f}")
    
    return results


def generate_forecast(results, daily, exog, forecast_days=30):
    """Generate forecast with confidence intervals."""
    print(f"\n🔮 Generating {forecast_days}-day forecast...")
    
    # Create future dates
    last_date = daily.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                  periods=forecast_days, freq='D')
    
    # Create future exogenous variables
    future_exog = pd.DataFrame(index=future_dates)
    
    # Day of week
    for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']):
        future_exog[day] = (future_dates.dayofweek == i).astype(int)
    
    future_exog['is_weekend'] = (future_dates.dayofweek >= 5).astype(int)
    
    # Month dummies
    for i in range(1, 12):
        future_exog[f'month_{i}'] = (future_dates.month == i).astype(int)
    
    # Quarter
    future_exog['Q1'] = (future_dates.quarter == 1).astype(int)
    future_exog['Q2'] = (future_dates.quarter == 2).astype(int)
    future_exog['Q3'] = (future_dates.quarter == 3).astype(int)
    
    # Month start/end
    future_exog['month_start'] = future_dates.is_month_start.astype(int)
    future_exog['month_end'] = future_dates.is_month_end.astype(int)
    
    # Trend (continue from training data)
    future_exog['trend'] = np.arange(len(daily), len(daily) + forecast_days)
    
    # Ensure columns match
    future_exog = future_exog[exog.columns]
    
    # Generate forecast
    forecast = results.get_forecast(steps=forecast_days, exog=future_exog)
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'predicted_sales': forecast.predicted_mean.values,
        'lower_ci': forecast.conf_int()['lower sales'].values,
        'upper_ci': forecast.conf_int()['upper sales'].values
    })
    
    # Floor at 0
    forecast_df['predicted_sales'] = forecast_df['predicted_sales'].clip(lower=0)
    forecast_df['lower_ci'] = forecast_df['lower_ci'].clip(lower=0)
    
    return forecast_df


def create_sarimax_visualization(daily, forecast_df, results):
    """Create SARIMAX forecast visualization."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Main forecast with CI
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Historical (last 90 days for clarity)
    recent = daily.tail(90)
    ax1.plot(recent.index, recent['sales'], color='#2c3e50', alpha=0.7, 
             linewidth=1, label='Historical')
    
    # Forecast
    ax1.plot(forecast_df['date'], forecast_df['predicted_sales'], 
             color='#e74c3c', linewidth=2, label='SARIMAX Forecast')
    
    # Confidence interval
    ax1.fill_between(forecast_df['date'], forecast_df['lower_ci'], forecast_df['upper_ci'],
                     color='#e74c3c', alpha=0.2, label='95% CI')
    
    ax1.axvline(daily.index.max(), color='#3498db', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Sales ($)')
    ax1.set_title('SARIMAX Sales Forecast with Confidence Interval', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax1.grid(True, alpha=0.3)
    
    # 2. Seasonal Decomposition
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Weekly seasonality from model (approximate)
    weekly_pattern = daily.groupby(daily.index.dayofweek)['sales'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    colors = ['#27ae60' if v > weekly_pattern.mean() else '#e74c3c' for v in weekly_pattern.values]
    
    ax2.bar(range(7), weekly_pattern.values, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(weekly_pattern.mean(), color='#2c3e50', linestyle='--', alpha=0.7,
                label=f'Avg: ${weekly_pattern.mean():,.0f}')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(days)
    ax2.set_ylabel('Average Daily Sales ($)')
    ax2.set_title('Weekly Seasonal Pattern', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Residual diagnostics
    ax3 = fig.add_subplot(2, 2, 3)
    
    residuals = results.resid
    ax3.plot(residuals.index, residuals, color='#3498db', alpha=0.5, linewidth=0.5)
    ax3.axhline(0, color='#e74c3c', linestyle='-', linewidth=1)
    ax3.axhline(residuals.std() * 2, color='#e74c3c', linestyle='--', alpha=0.5)
    ax3.axhline(-residuals.std() * 2, color='#e74c3c', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Residual ($)')
    ax3.set_title('Model Residuals (±2σ bounds)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Forecast detail
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Bar chart of forecast
    x = range(len(forecast_df))
    ax4.bar(x, forecast_df['predicted_sales'], color='#3498db', alpha=0.8, 
            edgecolor='white', label='Forecast')
    
    # Error bars for CI
    errors = [forecast_df['predicted_sales'] - forecast_df['lower_ci'],
              forecast_df['upper_ci'] - forecast_df['predicted_sales']]
    
    ax4.errorbar(x, forecast_df['predicted_sales'], yerr=errors, 
                 fmt='none', color='#2c3e50', capsize=2, alpha=0.5)
    
    # Labels
    labels = [d.strftime('%m/%d') for d in forecast_df['date']]
    ax4.set_xticks(range(0, len(labels), 5))
    ax4.set_xticklabels([labels[i] for i in range(0, len(labels), 5)], rotation=45)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Predicted Sales ($)')
    ax4.set_title('30-Day Forecast Detail with CI', fontsize=12, fontweight='bold')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def generate_sarimax_report(daily, forecast_df, results):
    """Generate SARIMAX forecast report."""
    
    report = f"""
{'='*80}
                    SARIMAX SALES FORECAST REPORT
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

📊 MODEL SUMMARY
{'─'*80}
Model Type:           SARIMAX (Seasonal ARIMA with Exogenous Variables)
Training Period:      {daily.index.min().strftime('%Y-%m-%d')} to {daily.index.max().strftime('%Y-%m-%d')}
Training Samples:     {len(daily):,}
AIC:                  {results.aic:.2f}
BIC:                  {results.bic:.2f}

Historical Stats:
  Mean Daily Sales:   ${daily['sales'].mean():>12,.0f}
  Median:             ${daily['sales'].median():>12,.0f}
  Std Deviation:      ${daily['sales'].std():>12,.0f}

📈 FORECAST RESULTS
{'─'*80}
Forecast Period:      {forecast_df['date'].min().strftime('%Y-%m-%d')} to {forecast_df['date'].max().strftime('%Y-%m-%d')}
Days Forecasted:      {len(forecast_df)}

30-Day Predictions:
  Total:              ${forecast_df['predicted_sales'].sum():>12,.0f}
  Daily Average:      ${forecast_df['predicted_sales'].mean():>12,.0f}
  
95% Confidence Interval:
  Lower Bound Total:  ${forecast_df['lower_ci'].sum():>12,.0f}
  Upper Bound Total:  ${forecast_df['upper_ci'].sum():>12,.0f}

📅 DAILY FORECAST BREAKDOWN
{'─'*80}
{'Date':<12} {'Day':<5} {'Forecast':>12} {'Lower CI':>12} {'Upper CI':>12}
{'─'*80}
"""
    
    for _, row in forecast_df.iterrows():
        dow = row['date'].strftime('%a')
        report += f"{row['date'].strftime('%Y-%m-%d'):<12} {dow:<5} ${row['predicted_sales']:>10,.0f} ${row['lower_ci']:>10,.0f} ${row['upper_ci']:>10,.0f}\n"
    
    report += f"""
📊 WEEKLY PATTERN
{'─'*80}
"""
    
    weekly = daily.groupby(daily.index.dayofweek)['sales'].mean()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, day in enumerate(days):
        report += f"  {day:<10}: ${weekly[i]:>10,.0f} avg/day\n"
    
    report += f"""
{'='*80}
                              END OF REPORT
{'='*80}
"""
    
    return report


def main():
    """Main SARIMAX forecasting pipeline."""
    
    print("="*80)
    print("   SARIMAX SALES FORECASTING")
    print("="*80)
    
    # Load data
    daily = load_and_prepare_data()
    
    # Analyze stationarity
    is_stationary = analyze_stationarity(daily['sales'])
    
    # Create exogenous features
    print("\n📊 Creating exogenous features...")
    exog = create_exogenous_features(daily)
    print(f"   Features: {len(exog.columns)}")
    
    # Fit SARIMAX model
    # Using weekly seasonality (7 days)
    # Order: (p, d, q) = (1, 1, 1) - autoregressive, differencing, moving average
    # Seasonal Order: (P, D, Q, s) = (1, 1, 1, 7) - weekly seasonality
    results = fit_sarimax_model(daily, exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    
    # Generate forecast
    forecast_df = generate_forecast(results, daily, exog, forecast_days=30)
    
    # Generate outputs
    print("\n💾 Saving outputs...")
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    fig = create_sarimax_visualization(daily, forecast_df, results)
    fig.savefig(REPORTS_PATH / "sarimax_forecast.png", dpi=150, bbox_inches='tight')
    print(f"   Plot: {REPORTS_PATH / 'sarimax_forecast.png'}")
    
    # Forecast CSV
    forecast_df.to_csv(REPORTS_PATH / "sarimax_forecast.csv", index=False)
    print(f"   Forecast data: {REPORTS_PATH / 'sarimax_forecast.csv'}")
    
    # Report
    report = generate_sarimax_report(daily, forecast_df, results)
    with open(REPORTS_PATH / "sarimax_forecast_report.txt", 'w') as f:
        f.write(report)
    print(f"   Report: {REPORTS_PATH / 'sarimax_forecast_report.txt'}")
    
    # Print summary
    print("\n" + "="*80)
    print("   📈 SARIMAX 30-DAY FORECAST SUMMARY")
    print("="*80)
    print(f"   Period:          {forecast_df['date'].min().strftime('%Y-%m-%d')} to {forecast_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Predicted Total: ${forecast_df['predicted_sales'].sum():>12,.0f}")
    print(f"   Daily Average:   ${forecast_df['predicted_sales'].mean():>12,.0f}")
    print(f"   95% CI Range:    ${forecast_df['lower_ci'].sum():>10,.0f} - ${forecast_df['upper_ci'].sum():,.0f}")
    print("="*80)
    
    print("\n✅ SARIMAX forecasting complete!")
    
    return results, forecast_df, daily


if __name__ == "__main__":
    results, forecast_df, daily = main()

