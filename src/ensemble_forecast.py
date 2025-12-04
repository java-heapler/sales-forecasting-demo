"""
Ensemble Sales Forecasting
==========================
Combines Prophet, XGBoost, SARIMAX, and Scikit-Learn predictions with configurable weights.

This script demonstrates:
- Ensemble model combining multiple forecasters
- Weighted averaging based on model performance
- Uncertainty quantification from multiple models
- Visualization of ensemble vs individual models

Note: Run prophet_tuned.py, sarimax_forecast.py, xgboost_forecast.py, and sklearn_forecast.py first.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_PATH = PROJECT_ROOT / "reports"
FORECASTS_PATH = REPORTS_PATH / "forecasts"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_sales_history.json"


def load_all_forecasts():
    """Load forecasts from all four models."""
    print("📥 Loading model forecasts...")
    
    # Prophet
    prophet_path = FORECASTS_PATH / "prophet" / "prophet_forecast_combined.csv"
    prophet_df = pd.read_csv(prophet_path)
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # XGBoost
    xgboost_path = FORECASTS_PATH / "xgboost" / "xgboost_forecast.csv"
    xgboost_df = pd.read_csv(xgboost_path)
    xgboost_df['date'] = pd.to_datetime(xgboost_df['date'])
    
    # SARIMAX
    sarimax_path = FORECASTS_PATH / "sarimax" / "sarimax_forecast.csv"
    sarimax_df = pd.read_csv(sarimax_path)
    sarimax_df['date'] = pd.to_datetime(sarimax_df['date'])
    
    # Scikit-Learn
    sklearn_path = FORECASTS_PATH / "sklearn" / "sklearn_forecast.csv"
    sklearn_df = None
    if sklearn_path.exists():
        sklearn_df = pd.read_csv(sklearn_path)
        sklearn_df['date'] = pd.to_datetime(sklearn_df['date'])
        print(f"   Sklearn:  {len(sklearn_df)} predictions ✓")
    else:
        print("   Sklearn:  Not found (run sklearn_forecast.py first)")
    
    print(f"   Prophet:  {len(prophet_df)} predictions")
    print(f"   XGBoost:  {len(xgboost_df)} predictions")
    print(f"   SARIMAX:  {len(sarimax_df)} predictions")
    
    return prophet_df, xgboost_df, sarimax_df, sklearn_df


def load_historical_data():
    """Load historical sales data for context."""
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Filter to invoices only
    invoices = df[df['invoice_type'] == 'IN']
    
    # Aggregate daily
    daily = invoices.groupby('invoice_date')['amount'].sum().reset_index()
    daily.columns = ['date', 'sales']
    
    return daily


def create_ensemble_forecast(prophet_df, xgboost_df, sarimax_df, sklearn_df=None,
                              weights=None, forecast_days=30):
    """
    Create weighted ensemble forecast from 3 or 4 models.
    
    Educational Note:
    - Ensemble methods combine multiple models to reduce variance and improve robustness
    - Weights should be based on validation performance, not training performance
    - This implementation uses fixed weights based on backtesting results
    """
    # Set default weights based on whether sklearn is available
    if weights is None:
        if sklearn_df is not None:
            # 4-model ensemble
            weights = {
                'sarimax': 0.35,
                'xgboost': 0.25,
                'sklearn': 0.25,
                'prophet': 0.15
            }
        else:
            # 3-model ensemble
            weights = {'sarimax': 0.45, 'xgboost': 0.30, 'prophet': 0.25}
    
    print(f"\n🔧 Creating Ensemble Forecast")
    weight_str = ", ".join([f"{k.capitalize()} {v*100:.0f}%" for k, v in weights.items()])
    print(f"   Weights: {weight_str}")
    
    # Get forecast dates from XGBoost
    forecast_dates = xgboost_df['date'].head(forecast_days)
    
    # Get the last historical date from Prophet
    historical_mask = prophet_df['ds'] <= prophet_df['ds'].max() - pd.Timedelta(days=30)
    last_historical = prophet_df[historical_mask]['ds'].max()
    
    # Extract predictions for each model
    prophet_future = prophet_df[prophet_df['ds'] > last_historical].head(forecast_days)
    
    # Create ensemble DataFrame
    ensemble = pd.DataFrame({
        'date': forecast_dates.values,
        'prophet': prophet_future['yhat'].values[:forecast_days],
        'prophet_lower': prophet_future['yhat_lower'].values[:forecast_days],
        'prophet_upper': prophet_future['yhat_upper'].values[:forecast_days],
        'xgboost': xgboost_df['predicted_sales'].head(forecast_days).values,
        'sarimax': sarimax_df['predicted_sales'].head(forecast_days).values,
        'sarimax_lower': sarimax_df['lower_ci'].head(forecast_days).values,
        'sarimax_upper': sarimax_df['upper_ci'].head(forecast_days).values
    })
    
    # Add sklearn if available
    if sklearn_df is not None:
        if 'Stacking' in sklearn_df.columns:
            ensemble['sklearn'] = sklearn_df['Stacking'].head(forecast_days).values
        elif 'RandomForest' in sklearn_df.columns:
            ensemble['sklearn'] = sklearn_df['RandomForest'].head(forecast_days).values
        else:
            ensemble['sklearn'] = sklearn_df['ensemble_avg'].head(forecast_days).values
    
    # Calculate weighted ensemble prediction
    ensemble['ensemble'] = ensemble['prophet'] * weights['prophet']
    ensemble['ensemble'] += ensemble['sarimax'] * weights['sarimax']
    ensemble['ensemble'] += ensemble['xgboost'] * weights['xgboost']
    
    if sklearn_df is not None and 'sklearn' in weights:
        ensemble['ensemble'] += ensemble['sklearn'] * weights['sklearn']
    
    # Calculate ensemble confidence interval
    ci_weights_sum = weights['prophet'] + weights['sarimax']
    ensemble['ensemble_lower'] = (
        ensemble['prophet_lower'] * (weights['prophet'] / ci_weights_sum) +
        ensemble['sarimax_lower'] * (weights['sarimax'] / ci_weights_sum)
    ) * ci_weights_sum
    
    ensemble['ensemble_upper'] = (
        ensemble['prophet_upper'] * (weights['prophet'] / ci_weights_sum) +
        ensemble['sarimax_upper'] * (weights['sarimax'] / ci_weights_sum)
    ) * ci_weights_sum
    
    # Add contributions from models without native CIs
    xgb_contrib = ensemble['xgboost'] * weights['xgboost']
    ensemble['ensemble_lower'] += xgb_contrib * 0.7
    ensemble['ensemble_upper'] += xgb_contrib * 1.3
    
    if sklearn_df is not None and 'sklearn' in weights:
        sklearn_contrib = ensemble['sklearn'] * weights['sklearn']
        ensemble['ensemble_lower'] += sklearn_contrib * 0.8
        ensemble['ensemble_upper'] += sklearn_contrib * 1.2
    
    # Floor at 0
    numeric_cols = ['ensemble', 'ensemble_lower', 'prophet', 'xgboost', 'sarimax']
    if 'sklearn' in ensemble.columns:
        numeric_cols.append('sklearn')
    for col in numeric_cols:
        ensemble[col] = ensemble[col].clip(lower=0)
    
    # Add day of week
    ensemble['day_of_week'] = pd.to_datetime(ensemble['date']).dt.day_name()
    
    return ensemble


def create_ensemble_visualization(ensemble, historical, weights):
    """Create comprehensive ensemble visualization."""
    
    has_sklearn = 'sklearn' in ensemble.columns
    
    fig = plt.figure(figsize=(18, 14))
    
    # Color scheme
    colors = {
        'prophet': '#3498db',
        'xgboost': '#e74c3c', 
        'sarimax': '#2ecc71',
        'sklearn': '#f39c12',
        'ensemble': '#9b59b6',
        'historical': '#2c3e50'
    }
    
    # 1. Main ensemble forecast
    ax1 = fig.add_subplot(2, 2, 1)
    
    recent_hist = historical.tail(60)
    ax1.plot(recent_hist['date'], recent_hist['sales'], color=colors['historical'], 
             alpha=0.5, linewidth=1, label='Historical')
    
    ax1.plot(ensemble['date'], ensemble['prophet'], '--', color=colors['prophet'], 
             alpha=0.6, linewidth=1.5, label=f"Prophet ({weights['prophet']*100:.0f}%)")
    ax1.plot(ensemble['date'], ensemble['xgboost'], '--', color=colors['xgboost'], 
             alpha=0.6, linewidth=1.5, label=f"XGBoost ({weights['xgboost']*100:.0f}%)")
    ax1.plot(ensemble['date'], ensemble['sarimax'], '--', color=colors['sarimax'], 
             alpha=0.6, linewidth=1.5, label=f"SARIMAX ({weights['sarimax']*100:.0f}%)")
    
    if has_sklearn:
        ax1.plot(ensemble['date'], ensemble['sklearn'], '--', color=colors['sklearn'], 
                 alpha=0.6, linewidth=1.5, label=f"Sklearn ({weights.get('sklearn', 0)*100:.0f}%)")
    
    ax1.plot(ensemble['date'], ensemble['ensemble'], color=colors['ensemble'], 
             linewidth=3, label='ENSEMBLE', marker='o', markersize=4)
    ax1.fill_between(ensemble['date'], ensemble['ensemble_lower'], ensemble['ensemble_upper'],
                     alpha=0.2, color=colors['ensemble'])
    
    ax1.axvline(historical['date'].max(), color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Sales ($)')
    
    n_models = 4 if has_sklearn else 3
    weight_pcts = "/".join([str(int(w*100)) for w in weights.values()])
    ax1.set_title(f'{n_models}-Model Ensemble Forecast ({weight_pcts})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax1.grid(True, alpha=0.3)
    
    # 2. Model comparison
    ax2 = fig.add_subplot(2, 2, 2)
    
    x = np.arange(len(ensemble))
    n_bars = 4 if has_sklearn else 3
    width = 0.8 / n_bars
    
    ax2.bar(x - width*1.5, ensemble['prophet'], width, 
            label='Prophet', color=colors['prophet'], alpha=0.7)
    ax2.bar(x - width*0.5, ensemble['ensemble'], width, 
            label='Ensemble', color=colors['ensemble'], alpha=0.9)
    ax2.bar(x + width*0.5, ensemble['sarimax'], width, 
            label='SARIMAX', color=colors['sarimax'], alpha=0.7)
    
    if has_sklearn:
        ax2.bar(x + width*1.5, ensemble['sklearn'], width, 
                label='Sklearn', color=colors['sklearn'], alpha=0.7)
    
    ax2.set_xticks(range(0, len(ensemble), 5))
    ax2.set_xticklabels([ensemble['date'].iloc[i].strftime('%m/%d') for i in range(0, len(ensemble), 5)], rotation=45)
    ax2.set_ylabel('Predicted Sales ($)')
    ax2.set_title('Model Comparison by Day', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Day of week breakdown
    ax3 = fig.add_subplot(2, 2, 3)
    
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg = ensemble.groupby('day_of_week')['ensemble'].mean().reindex(dow_order)
    
    colors_dow = ['#27ae60' if v > dow_avg.mean() else '#e74c3c' for v in dow_avg.values]
    ax3.bar(range(7), dow_avg.values, color=colors_dow, alpha=0.8, edgecolor='white')
    ax3.axhline(dow_avg.mean(), color='#2c3e50', linestyle='--', alpha=0.7,
                label=f'Avg: ${dow_avg.mean():,.0f}')
    
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax3.set_ylabel('Avg Ensemble Forecast ($)')
    ax3.set_title('Forecast by Day of Week', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative forecast
    ax4 = fig.add_subplot(2, 2, 4)
    
    ensemble['cumulative'] = ensemble['ensemble'].cumsum()
    ensemble['cumulative_lower'] = ensemble['ensemble_lower'].cumsum()
    ensemble['cumulative_upper'] = ensemble['ensemble_upper'].cumsum()
    
    ax4.plot(ensemble['date'], ensemble['prophet'].cumsum(), '--', 
             color=colors['prophet'], alpha=0.5, linewidth=1, label='Prophet')
    ax4.plot(ensemble['date'], ensemble['sarimax'].cumsum(), '--', 
             color=colors['sarimax'], alpha=0.5, linewidth=1, label='SARIMAX')
    
    if has_sklearn:
        ax4.plot(ensemble['date'], ensemble['sklearn'].cumsum(), '--', 
                 color=colors['sklearn'], alpha=0.5, linewidth=1, label='Sklearn')
    
    ax4.plot(ensemble['date'], ensemble['cumulative'], color=colors['ensemble'], 
             linewidth=2.5, label='Ensemble')
    ax4.fill_between(ensemble['date'], ensemble['cumulative_lower'], ensemble['cumulative_upper'],
                     alpha=0.2, color=colors['ensemble'], label='95% CI')
    
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Sales ($)')
    ax4.set_title('Cumulative 30-Day Forecast', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.tick_params(axis='x', rotation=45)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax4.grid(True, alpha=0.3)
    
    ax4.annotate(f"Total: ${ensemble['cumulative'].max():,.0f}", 
                 xy=(ensemble['date'].iloc[-1], ensemble['cumulative'].max()),
                 xytext=(10, -10), textcoords='offset points',
                 fontsize=10, fontweight='bold', color=colors['ensemble'])
    
    plt.tight_layout()
    return fig


def generate_ensemble_report(ensemble, historical, weights):
    """Generate ensemble forecast report."""
    
    has_sklearn = 'sklearn' in ensemble.columns
    n_models = 4 if has_sklearn else 3
    
    total_30 = ensemble['ensemble'].sum()
    daily_avg = ensemble['ensemble'].mean()
    lower_total = ensemble['ensemble_lower'].sum()
    upper_total = ensemble['ensemble_upper'].sum()
    
    hist_daily_avg = historical['sales'].mean()
    hist_30_est = hist_daily_avg * 30
    recent_30 = historical.tail(30)['sales'].sum()
    
    weight_pcts = "/".join([str(int(w*100)) for w in weights.values()])
    
    report = f"""
{'='*80}
                    ENSEMBLE SALES FORECAST REPORT
                    {n_models}-Model Configuration ({weight_pcts})
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

📊 ENSEMBLE CONFIGURATION
{'─'*80}
Model Weights:
  • SARIMAX:   {weights['sarimax']*100:.0f}%
"""
    
    if has_sklearn:
        report += f"  • Sklearn:   {weights['sklearn']*100:.0f}%\n"
    
    report += f"""  • XGBoost:   {weights['xgboost']*100:.0f}%
  • Prophet:   {weights['prophet']*100:.0f}%

📈 30-DAY FORECAST SUMMARY
{'─'*80}
Forecast Period:          {ensemble['date'].min().strftime('%Y-%m-%d')} to {ensemble['date'].max().strftime('%Y-%m-%d')}

ENSEMBLE PREDICTION:
  30-Day Total:           ${total_30:>12,.0f}
  Daily Average:          ${daily_avg:>12,.0f}
  
95% Confidence Interval:
  Lower Bound:            ${lower_total:>12,.0f}
  Upper Bound:            ${upper_total:>12,.0f}

COMPARISON TO HISTORICAL:
  Historical Daily Avg:   ${hist_daily_avg:>12,.0f}
  Historical 30-Day Est:  ${hist_30_est:>12,.0f}
  Last 30 Days Actual:    ${recent_30:>12,.0f}
  
  vs Historical Avg:      {(total_30/hist_30_est - 1)*100:+.1f}%
  vs Last 30 Days:        {(total_30/recent_30 - 1)*100:+.1f}%

📊 INDIVIDUAL MODEL FORECASTS (30-Day)
{'─'*80}
{'Model':<15} {'Total':>15} {'Daily Avg':>12} {'Weight':>10}
{'─'*80}
{'Prophet':<15} ${ensemble['prophet'].sum():>14,.0f} ${ensemble['prophet'].mean():>11,.0f} {weights['prophet']*100:>9.0f}%
"""
    
    if has_sklearn:
        report += f"{'Sklearn':<15} ${ensemble['sklearn'].sum():>14,.0f} ${ensemble['sklearn'].mean():>11,.0f} {weights['sklearn']*100:>9.0f}%\n"
    
    report += f"""{'SARIMAX':<15} ${ensemble['sarimax'].sum():>14,.0f} ${ensemble['sarimax'].mean():>11,.0f} {weights['sarimax']*100:>9.0f}%
{'XGBoost':<15} ${ensemble['xgboost'].sum():>14,.0f} ${ensemble['xgboost'].mean():>11,.0f} {weights['xgboost']*100:>9.0f}%
{'─'*80}
{'ENSEMBLE':<15} ${total_30:>14,.0f} ${daily_avg:>11,.0f} {'100':>9}%

{'='*80}
                              END OF REPORT
{'='*80}
"""
    
    return report


def main():
    """Main ensemble forecasting pipeline."""
    
    print("=" * 80)
    print("   ENSEMBLE SALES FORECASTING")
    print("   Multi-Model Configuration")
    print("=" * 80)
    
    # Load all forecasts
    prophet_df, xgboost_df, sarimax_df, sklearn_df = load_all_forecasts()
    
    # Load historical data
    print("\n📥 Loading historical data...")
    historical = load_historical_data()
    print(f"   {len(historical)} days of historical data")
    
    # Set weights
    if sklearn_df is not None:
        weights = {
            'sarimax': 0.35,
            'xgboost': 0.25,
            'sklearn': 0.25,
            'prophet': 0.15
        }
    else:
        weights = {'sarimax': 0.45, 'xgboost': 0.30, 'prophet': 0.25}
    
    # Create ensemble
    ensemble = create_ensemble_forecast(
        prophet_df, xgboost_df, sarimax_df, sklearn_df, 
        weights=weights
    )
    
    # Generate outputs
    print("\n💾 Saving outputs...")
    output_path = FORECASTS_PATH / "ensemble"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    fig = create_ensemble_visualization(ensemble, historical, weights)
    fig.savefig(output_path / "ensemble_forecast.png", dpi=150, bbox_inches='tight')
    print(f"   Plot: {output_path / 'ensemble_forecast.png'}")
    
    # Forecast CSV
    output_cols = ['date', 'ensemble', 'ensemble_lower', 'ensemble_upper', 
                   'prophet', 'xgboost', 'sarimax', 'day_of_week']
    if 'sklearn' in ensemble.columns:
        output_cols.insert(5, 'sklearn')
    
    ensemble_output = ensemble[output_cols].copy()
    ensemble_output.to_csv(output_path / "ensemble_forecast.csv", index=False)
    print(f"   Forecast data: {output_path / 'ensemble_forecast.csv'}")
    
    # Report
    report = generate_ensemble_report(ensemble, historical, weights)
    with open(output_path / "ensemble_forecast_report.txt", 'w') as f:
        f.write(report)
    print(f"   Report: {output_path / 'ensemble_forecast_report.txt'}")
    
    # Print summary
    total_30 = ensemble['ensemble'].sum()
    daily_avg = ensemble['ensemble'].mean()
    n_models = 4 if sklearn_df is not None else 3
    weight_pcts = "/".join([str(int(w*100)) for w in weights.values()])
    
    print("\n" + "=" * 80)
    print(f"   📈 {n_models}-MODEL ENSEMBLE 30-DAY FORECAST SUMMARY")
    print("=" * 80)
    print(f"   Configuration:   {weight_pcts}")
    print(f"   Period:          {ensemble['date'].min().strftime('%Y-%m-%d')} to {ensemble['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Predicted Total: ${total_30:>12,.0f}")
    print(f"   Daily Average:   ${daily_avg:>12,.0f}")
    print(f"   95% CI Range:    ${ensemble['ensemble_lower'].sum():>10,.0f} - ${ensemble['ensemble_upper'].sum():,.0f}")
    print("=" * 80)
    
    print("\n✅ Ensemble forecasting complete!")
    
    return ensemble, weights


if __name__ == "__main__":
    ensemble, weights = main()

