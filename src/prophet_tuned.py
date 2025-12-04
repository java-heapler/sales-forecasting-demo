"""
Prophet Tuning with Train/Test Validation
==========================================
Simplified hyperparameter tuning using train/test split.

This script demonstrates:
- Prophet time series forecasting
- Hyperparameter tuning via grid search
- Train/test validation for time series
- Handling overprediction bias

Educational Note:
Prophet is Facebook's forecasting tool designed for business time series.
It handles seasonality (weekly, yearly) and holidays automatically.
"""

import json
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_sales_history.json"
REPORTS_PATH = PROJECT_ROOT / "reports" / "forecasts" / "prophet"


def load_data():
    """Load and prepare sales data for Prophet."""
    print("📥 Loading sales data...")
    
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Filter to invoices only
    df = df[df['invoice_type'] == 'IN'].copy()
    
    # Aggregate daily
    daily = df.groupby('invoice_date')['amount'].sum().reset_index()
    daily.columns = ['ds', 'y']
    
    # Fill missing dates
    date_range = pd.date_range(start=daily['ds'].min(), end=daily['ds'].max(), freq='D')
    daily = daily.set_index('ds').reindex(date_range, fill_value=0).reset_index()
    daily.columns = ['ds', 'y']
    
    print(f"   Date range: {daily['ds'].min().strftime('%Y-%m-%d')} to {daily['ds'].max().strftime('%Y-%m-%d')}")
    print(f"   Total days: {len(daily)}")
    print(f"   Daily avg: ${daily['y'].mean():,.0f}")
    
    return daily


def train_test_split_ts(df, test_days=30):
    """Split data for time series validation."""
    cutoff = df['ds'].max() - timedelta(days=test_days)
    train = df[df['ds'] <= cutoff].copy()
    test = df[df['ds'] > cutoff].copy()
    return train, test


def evaluate_prophet_params(train, test, params):
    """Train Prophet with given params and evaluate on test set."""
    
    model = Prophet(
        growth=params.get('growth', 'linear'),
        changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=params.get('seasonality_prior_scale', 1.0),
        seasonality_mode=params.get('seasonality_mode', 'multiplicative'),
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )
    
    model.add_country_holidays(country_name='US')
    model.fit(train)
    
    # Predict on test dates
    future = pd.DataFrame({'ds': test['ds']})
    forecast = model.predict(future)
    
    # Calculate metrics
    predictions = forecast['yhat'].clip(lower=0).values
    actuals = test['y'].values
    
    mae = np.abs(predictions - actuals).mean()
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # Bias (total prediction error %)
    total_pred = predictions.sum()
    total_actual = actuals.sum()
    bias_pct = (total_pred - total_actual) / total_actual * 100
    
    # MAPE (avoiding division by zero)
    mask = actuals > 0
    if mask.sum() > 0:
        mape = np.abs((predictions[mask] - actuals[mask]) / actuals[mask]).mean() * 100
    else:
        mape = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'bias_pct': bias_pct,
        'total_predicted': total_pred,
        'total_actual': total_actual
    }, model, forecast


def grid_search_prophet(train, test, param_grid):
    """Grid search over Prophet hyperparameters."""
    
    print("\n🔍 Starting Prophet Grid Search...")
    print(f"   Test period: {test['ds'].min().strftime('%Y-%m-%d')} to {test['ds'].max().strftime('%Y-%m-%d')}")
    print(f"   Test actual: ${test['y'].sum():,.0f}")
    
    results = []
    best_model = None
    best_forecast = None
    best_mae = float('inf')
    
    # Generate all combinations
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"   Combinations: {len(combinations)}")
    print("\n" + "─" * 70)
    print(f"{'#':<4} {'CPS':>6} {'SPS':>6} {'Mode':>6} {'MAE':>12} {'Bias':>10}")
    print("─" * 70)
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        
        try:
            metrics, model, forecast = evaluate_prophet_params(train, test, params)
            
            mode_short = params['seasonality_mode'][:4]
            print(f"{i:<4} {params['changepoint_prior_scale']:>6.3f} {params['seasonality_prior_scale']:>6.1f} {mode_short:>6} ${metrics['mae']:>10,.0f} {metrics['bias_pct']:>+9.1f}%")
            
            results.append({
                **params,
                **metrics
            })
            
            # Track best model
            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_model = model
                best_forecast = forecast
                
        except Exception as e:
            print(f"{i:<4} Error: {str(e)[:40]}")
    
    print("─" * 70)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mae')
    
    return results_df, best_model, best_forecast


def train_final_model(df, best_params):
    """Train final model on all data with best parameters."""
    
    print(f"\n🎓 Training final model on all data...")
    
    model = Prophet(
        growth=best_params.get('growth', 'linear'),
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        seasonality_mode=best_params['seasonality_mode'],
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )
    
    model.add_country_holidays(country_name='US')
    model.fit(df)
    
    # Generate 90-day forecast
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    
    return model, forecast


def create_visualization(df, tuning_results, final_forecast, train, test, test_forecast):
    """Create tuning visualization."""
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Prophet Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
    
    # 1. MAE by parameters
    ax1 = fig.add_subplot(2, 3, 1)
    
    top_10 = tuning_results.head(10)
    y_pos = np.arange(len(top_10))
    colors = ['#27ae60' if i == 0 else '#3498db' for i in range(len(top_10))]
    
    ax1.barh(y_pos, top_10['mae'], color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    labels = [f"cps={r['changepoint_prior_scale']:.3f}, sps={r['seasonality_prior_scale']:.1f}" 
              for _, r in top_10.iterrows()]
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('MAE ($)')
    ax1.set_title('Top 10 Parameter Combinations (by MAE)')
    ax1.invert_yaxis()
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Bias comparison
    ax2 = fig.add_subplot(2, 3, 2)
    
    colors_bias = ['#27ae60' if abs(b) < 15 else '#f39c12' if abs(b) < 30 else '#e74c3c' 
                   for b in top_10['bias_pct']]
    ax2.barh(y_pos, top_10['bias_pct'], color=colors_bias, alpha=0.8)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(72, color='#e74c3c', linestyle='--', alpha=0.5, label='Original (+72%)')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Prediction Bias (%)')
    ax2.set_title('Bias Comparison (vs Original +72%)')
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Test set predictions vs actual
    ax3 = fig.add_subplot(2, 3, 3)
    
    ax3.plot(test['ds'], test['y'], color='#2c3e50', linewidth=2, label='Actual', marker='o', markersize=3)
    ax3.plot(test['ds'], test_forecast['yhat'].clip(lower=0), color='#27ae60', 
             linewidth=2, linestyle='--', label='Tuned Prediction')
    ax3.fill_between(test['ds'], 
                     test_forecast['yhat_lower'].clip(lower=0),
                     test_forecast['yhat_upper'].clip(lower=0),
                     alpha=0.2, color='#27ae60')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sales ($)')
    ax3.set_title('Test Set: Actual vs Tuned Prediction')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax3.grid(True, alpha=0.3)
    
    # 4. Full historical + forecast
    ax4 = fig.add_subplot(2, 3, 4)
    
    ax4.plot(df['ds'], df['y'], color='#2c3e50', alpha=0.5, linewidth=0.8, label='Historical')
    
    future_mask = final_forecast['ds'] > df['ds'].max()
    ax4.plot(final_forecast.loc[future_mask, 'ds'], final_forecast.loc[future_mask, 'yhat'],
             color='#27ae60', linewidth=2, label='Tuned Forecast')
    ax4.fill_between(final_forecast.loc[future_mask, 'ds'],
                     final_forecast.loc[future_mask, 'yhat_lower'],
                     final_forecast.loc[future_mask, 'yhat_upper'],
                     alpha=0.2, color='#27ae60')
    
    ax4.axvline(df['ds'].max(), color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Daily Sales ($)')
    ax4.set_title('Tuned Prophet: Full Forecast')
    ax4.legend(loc='upper left')
    ax4.tick_params(axis='x', rotation=45)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax4.grid(True, alpha=0.3)
    
    # 5. 30-Day forecast detail
    ax5 = fig.add_subplot(2, 3, 5)
    
    future_30 = final_forecast[final_forecast['ds'] > df['ds'].max()].head(30)
    
    ax5.bar(range(len(future_30)), future_30['yhat'], color='#27ae60', alpha=0.8, edgecolor='white')
    ax5.set_xticks(range(0, 30, 5))
    ax5.set_xticklabels([future_30['ds'].iloc[i].strftime('%m/%d') for i in range(0, 30, 5)], rotation=45)
    ax5.set_ylabel('Predicted Sales ($)')
    ax5.set_title('30-Day Tuned Forecast')
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax5.grid(True, alpha=0.3, axis='y')
    
    avg = future_30['yhat'].mean()
    ax5.axhline(avg, color='#e74c3c', linestyle='--', label=f'Avg: ${avg:,.0f}')
    ax5.legend()
    
    # 6. Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    best = tuning_results.iloc[0]
    future_30 = final_forecast[final_forecast['ds'] > df['ds'].max()].head(30)
    
    summary_text = f"""
    PROPHET TUNING SUMMARY
    ══════════════════════════════════════════════
    
    ORIGINAL PARAMETERS (had +72% bias):
    • changepoint_prior_scale: 0.1
    • seasonality_prior_scale: 10
    • seasonality_mode: multiplicative
    
    TUNED PARAMETERS (best by MAE):
    • changepoint_prior_scale: {best['changepoint_prior_scale']:.3f}
    • seasonality_prior_scale: {best['seasonality_prior_scale']:.1f}
    • seasonality_mode: {best['seasonality_mode']}
    
    VALIDATION METRICS:
    • Test MAE: ${best['mae']:,.0f}
    • Test Bias: {best['bias_pct']:+.1f}%  (was +72%)
    • Improvement: {72 - best['bias_pct']:.1f}% less bias
    
    30-DAY FORECAST (Tuned):
    • Total: ${future_30['yhat'].sum():,.0f}
    • Daily Avg: ${future_30['yhat'].mean():,.0f}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    return fig


def generate_report(df, tuning_results, final_forecast, test):
    """Generate tuning report."""
    
    best = tuning_results.iloc[0]
    future_30 = final_forecast[final_forecast['ds'] > df['ds'].max()].head(30)
    
    report = f"""
{'='*80}
                    PROPHET HYPERPARAMETER TUNING REPORT
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

📊 TUNING SUMMARY
{'─'*80}
Parameter combinations tested: {len(tuning_results)}
Test period: {test['ds'].min().strftime('%Y-%m-%d')} to {test['ds'].max().strftime('%Y-%m-%d')}
Test period actual sales: ${test['y'].sum():,.0f}

🏆 BEST PARAMETERS FOUND
{'─'*80}
changepoint_prior_scale:  {best['changepoint_prior_scale']:.3f}
seasonality_prior_scale:  {best['seasonality_prior_scale']:.1f}
seasonality_mode:         {best['seasonality_mode']}

📈 VALIDATION METRICS (Best Model)
{'─'*80}
MAE:               ${best['mae']:,.0f}
RMSE:              ${best['rmse']:,.0f}
Prediction Bias:   {best['bias_pct']:+.1f}%

⚖️ BIAS IMPROVEMENT
{'─'*80}
Original Bias:     +72.0%
Tuned Bias:        {best['bias_pct']:+.1f}%
Improvement:       {72 - best['bias_pct']:.1f} percentage points

📅 30-DAY FORECAST (Tuned Model)
{'─'*80}
Forecast Period:    {future_30['ds'].min().strftime('%Y-%m-%d')} to {future_30['ds'].max().strftime('%Y-%m-%d')}
Total:              ${future_30['yhat'].sum():>12,.0f}
Daily Average:      ${future_30['yhat'].mean():>12,.0f}
95% CI Lower:       ${future_30['yhat_lower'].sum():>12,.0f}
95% CI Upper:       ${future_30['yhat_upper'].sum():>12,.0f}

📊 ALL PARAMETER COMBINATIONS (sorted by MAE)
{'─'*80}
{'#':<4} {'CPS':>8} {'SPS':>8} {'Mode':>8} {'MAE':>12} {'Bias':>10}
{'─'*80}
"""
    
    for i, (_, row) in enumerate(tuning_results.iterrows(), 1):
        mode_short = row['seasonality_mode'][:5] if pd.notna(row['seasonality_mode']) else 'N/A'
        report += f"{i:<4} {row['changepoint_prior_scale']:>8.3f} {row['seasonality_prior_scale']:>8.1f} {mode_short:>8} ${row['mae']:>10,.0f} {row['bias_pct']:>+9.1f}%\n"
    
    report += f"""
💡 RECOMMENDATIONS
{'─'*80}
1. Use tuned parameters for future forecasts
2. Lower changepoint_prior_scale = more conservative trend
3. The original model was too aggressive in extrapolating growth
4. Re-run tuning quarterly as data patterns may shift

{'='*80}
                              END OF REPORT
{'='*80}
"""
    
    return report


def main():
    """Main Prophet tuning pipeline."""
    
    print("=" * 80)
    print("   PROPHET HYPERPARAMETER TUNING")
    print("   Goal: Reduce +72% overprediction bias")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Train/test split (last 30 days for testing)
    train, test = train_test_split_ts(df, test_days=30)
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(train)} days")
    print(f"   Test: {len(test)} days")
    
    # Define parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1],
        'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    # Grid search
    tuning_results, _, test_forecast = grid_search_prophet(train, test, param_grid)
    
    # Get best parameters
    best_params = tuning_results.iloc[0].to_dict()
    
    print("\n" + "=" * 80)
    print("   🏆 BEST PARAMETERS FOUND")
    print("=" * 80)
    print(f"   changepoint_prior_scale: {best_params['changepoint_prior_scale']}")
    print(f"   seasonality_prior_scale: {best_params['seasonality_prior_scale']}")
    print(f"   seasonality_mode: {best_params['seasonality_mode']}")
    print(f"   Test MAE: ${best_params['mae']:,.0f}")
    print(f"   Test Bias: {best_params['bias_pct']:+.1f}% (was +72%)")
    
    # Train final model on all data
    final_model, final_forecast = train_final_model(df, best_params)
    
    # Get test forecast for visualization
    test_future = pd.DataFrame({'ds': test['ds']})
    test_forecast = final_model.predict(test_future)
    
    # Generate outputs
    print("\n💾 Saving outputs...")
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    fig = create_visualization(df, tuning_results, final_forecast, train, test, test_forecast)
    fig.savefig(REPORTS_PATH / "prophet_tuning_results.png", dpi=150, bbox_inches='tight')
    print(f"   Plot: {REPORTS_PATH / 'prophet_tuning_results.png'}")
    
    # Tuning results CSV
    tuning_results.to_csv(REPORTS_PATH / "prophet_tuning_results.csv", index=False)
    print(f"   Tuning results: {REPORTS_PATH / 'prophet_tuning_results.csv'}")
    
    # Update main forecast file
    forecast_output = final_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'weekly', 'yearly']].copy()
    forecast_output.to_csv(REPORTS_PATH / "prophet_forecast_combined.csv", index=False)
    print(f"   Updated forecast: {REPORTS_PATH / 'prophet_forecast_combined.csv'}")
    
    # Report
    report = generate_report(df, tuning_results, final_forecast, test)
    with open(REPORTS_PATH / "prophet_tuning_report.txt", 'w') as f:
        f.write(report)
    print(f"   Report: {REPORTS_PATH / 'prophet_tuning_report.txt'}")
    
    # Print forecast summary
    future_30 = final_forecast[final_forecast['ds'] > df['ds'].max()].head(30)
    
    print("\n" + "=" * 80)
    print("   📈 TUNED PROPHET 30-DAY FORECAST")
    print("=" * 80)
    print(f"   Bias Improvement: +72% → {best_params['bias_pct']:+.1f}%")
    print(f"   Period: {future_30['ds'].min().strftime('%Y-%m-%d')} to {future_30['ds'].max().strftime('%Y-%m-%d')}")
    print(f"   Predicted Total: ${future_30['yhat'].sum():>12,.0f}")
    print(f"   Daily Average: ${future_30['yhat'].mean():>12,.0f}")
    print("=" * 80)
    
    print("\n✅ Prophet tuning complete!")
    
    return tuning_results, final_model, final_forecast


if __name__ == "__main__":
    tuning_results, final_model, final_forecast = main()

