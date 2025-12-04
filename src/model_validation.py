"""
Model Validation & Backtesting
==============================
Proper evaluation of all forecasting models using held-out test data.

This script:
1. Creates proper train/validation/test splits
2. Backtests all models on historical data
3. Computes true out-of-sample metrics
4. Identifies which models help vs hurt accuracy
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_sales_history.json"
REPORTS_PATH = PROJECT_ROOT / "reports" / "validation"


def load_data():
    """Load and prepare sales data."""
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
    daily.columns = ['date', 'sales']
    
    # Fill missing dates
    date_range = pd.date_range(start=daily['date'].min(), end=daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(date_range, fill_value=0).reset_index()
    daily.columns = ['date', 'sales']
    
    print(f"   Date range: {daily['date'].min().strftime('%Y-%m-%d')} to {daily['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Total days: {len(daily)}")
    print(f"   Daily avg: ${daily['sales'].mean():,.0f}")
    
    return daily


def create_train_test_split(daily, test_days=30):
    """
    Create proper train/test split for time series.
    
    We hold out the LAST test_days for true out-of-sample evaluation.
    """
    cutoff_date = daily['date'].max() - timedelta(days=test_days)
    
    train = daily[daily['date'] <= cutoff_date].copy()
    test = daily[daily['date'] > cutoff_date].copy()
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {train['date'].min().strftime('%Y-%m-%d')} to {train['date'].max().strftime('%Y-%m-%d')} ({len(train)} days)")
    print(f"   Test:  {test['date'].min().strftime('%Y-%m-%d')} to {test['date'].max().strftime('%Y-%m-%d')} ({len(test)} days)")
    
    return train, test


def create_ml_features(df, exclude_leaky=True):
    """Create features for ML models (XGBoost, sklearn)."""
    df = df.copy()
    
    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    
    # Cyclical encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features (CRITICAL: must shift to avoid leakage)
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    
    # Rolling features (must use shift to avoid leakage)
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df['sales'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df['sales'].shift(1).rolling(window).std()
    
    df['expanding_mean'] = df['sales'].shift(1).expanding().mean()
    
    return df


def evaluate_prophet(train, test):
    """Evaluate Prophet model on test set."""
    print("\n🔮 Evaluating Prophet...")
    
    # Prepare training data
    prophet_train = train[['date', 'sales']].copy()
    prophet_train.columns = ['ds', 'y']
    
    # Fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative'
    )
    model.add_country_holidays(country_name='US')
    model.fit(prophet_train)
    
    # Predict on test dates
    future = pd.DataFrame({'ds': test['date']})
    forecast = model.predict(future)
    
    # Get predictions
    predictions = forecast['yhat'].clip(lower=0).values
    actuals = test['sales'].values
    
    return predictions, actuals


def evaluate_sarimax(train, test):
    """Evaluate SARIMAX model on test set."""
    print("\n🔮 Evaluating SARIMAX...")
    
    # Prepare training data
    train_indexed = train.set_index('date')['sales']
    train_indexed = train_indexed.asfreq('D', fill_value=0)
    
    # Create simple exog (day of week only for stability)
    def create_exog(dates):
        exog = pd.DataFrame(index=dates)
        for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']):
            exog[day] = (dates.dayofweek == i).astype(int)
        return exog
    
    train_exog = create_exog(train_indexed.index)
    
    # Fit model with more conservative parameters
    try:
        model = SARIMAX(
            train_indexed,
            exog=train_exog,
            order=(1, 0, 1),  # Reduced differencing
            seasonal_order=(0, 1, 1, 7),  # Reduced seasonal AR
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False, maxiter=200)
        
        # Predict on test dates
        test_exog = create_exog(pd.DatetimeIndex(test['date']))
        forecast = results.get_forecast(steps=len(test), exog=test_exog)
        predictions = forecast.predicted_mean.clip(lower=0).values
        
    except Exception as e:
        print(f"   SARIMAX failed: {e}")
        predictions = np.full(len(test), train['sales'].mean())
    
    actuals = test['sales'].values
    
    return predictions, actuals


def evaluate_xgboost(train, test, fix_leakage=True):
    """
    Evaluate XGBoost model on test set.
    
    Args:
        fix_leakage: If True, exclude leaky features (invoice_count, customer_count)
    """
    print(f"\n🔮 Evaluating XGBoost (fix_leakage={fix_leakage})...")
    
    # Combine for feature creation
    all_data = pd.concat([train, test]).reset_index(drop=True)
    all_data = create_ml_features(all_data, exclude_leaky=fix_leakage)
    
    # Drop NaN from lag features
    all_data = all_data.dropna()
    
    # Split back
    cutoff = train['date'].max()
    train_feat = all_data[all_data['date'] <= cutoff]
    test_feat = all_data[all_data['date'] > cutoff]
    
    # Prepare features
    feature_cols = [c for c in train_feat.columns if c not in ['date', 'sales']]
    X_train = train_feat[feature_cols]
    y_train = train_feat['sales']
    X_test = test_feat[feature_cols]
    
    # Fit model
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test).clip(min=0)
    actuals = test_feat['sales'].values
    
    return predictions, actuals


def evaluate_sklearn_stacking(train, test):
    """Evaluate sklearn Stacking ensemble on test set."""
    print("\n🔮 Evaluating Sklearn Stacking...")
    
    # Combine for feature creation
    all_data = pd.concat([train, test]).reset_index(drop=True)
    all_data = create_ml_features(all_data, exclude_leaky=True)
    
    # Drop NaN
    all_data = all_data.dropna()
    
    # Split back
    cutoff = train['date'].max()
    train_feat = all_data[all_data['date'] <= cutoff]
    test_feat = all_data[all_data['date'] > cutoff]
    
    # Prepare features
    feature_cols = [c for c in train_feat.columns if c not in ['date', 'sales']]
    X_train = train_feat[feature_cols]
    y_train = train_feat['sales']
    X_test = test_feat[feature_cols]
    
    # Create stacking ensemble
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
        ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
    ]
    
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3
    )
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test).clip(min=0)
    actuals = test_feat['sales'].values
    
    return predictions, actuals


def evaluate_naive_baseline(train, test):
    """Naive baseline: predict last week's values."""
    print("\n🔮 Evaluating Naive Baseline (same day last week)...")
    
    predictions = []
    for test_date in test['date']:
        # Get same day from last week
        last_week_date = test_date - timedelta(days=7)
        last_week_value = train[train['date'] == last_week_date]['sales'].values
        
        if len(last_week_value) > 0:
            predictions.append(last_week_value[0])
        else:
            predictions.append(train['sales'].mean())
    
    actuals = test['sales'].values
    
    return np.array(predictions), actuals


def compute_metrics(predictions, actuals, model_name):
    """Compute evaluation metrics."""
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # MAPE (handle zeros)
    mask = actuals > 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(actuals[mask], predictions[mask]) * 100
    else:
        mape = np.nan
    
    # Total prediction vs actual
    total_pred = predictions.sum()
    total_actual = actuals.sum()
    total_error_pct = (total_pred - total_actual) / total_actual * 100
    
    return {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'total_predicted': total_pred,
        'total_actual': total_actual,
        'total_error_pct': total_error_pct
    }


def create_validation_visualization(results_df, all_predictions, test, train):
    """Create validation visualization."""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Model Comparison (MAE)
    ax1 = fig.add_subplot(2, 3, 1)
    sorted_results = results_df.sort_values('mae')
    colors = ['#27ae60' if r['mae'] == sorted_results['mae'].min() else 
              '#e74c3c' if r['mae'] == sorted_results['mae'].max() else '#3498db' 
              for _, r in sorted_results.iterrows()]
    ax1.barh(range(len(sorted_results)), sorted_results['mae'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(sorted_results)))
    ax1.set_yticklabels(sorted_results['model'])
    ax1.set_xlabel('Mean Absolute Error ($)')
    ax1.set_title('Model Comparison: MAE (lower is better)', fontweight='bold')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # 2. Total Prediction Error
    ax2 = fig.add_subplot(2, 3, 2)
    sorted_by_total = results_df.sort_values('total_error_pct', key=abs)
    colors = ['#27ae60' if abs(e) < 10 else '#f39c12' if abs(e) < 25 else '#e74c3c' 
              for e in sorted_by_total['total_error_pct']]
    bars = ax2.barh(range(len(sorted_by_total)), sorted_by_total['total_error_pct'], color=colors, alpha=0.8)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_yticks(range(len(sorted_by_total)))
    ax2.set_yticklabels(sorted_by_total['model'])
    ax2.set_xlabel('Total Prediction Error (%)')
    ax2.set_title('30-Day Total Error (closer to 0 is better)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # Add actual total annotation
    actual_total = results_df['total_actual'].iloc[0]
    ax2.annotate(f'Actual 30-Day: ${actual_total:,.0f}', xy=(0.95, 0.02), 
                xycoords='axes fraction', ha='right', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Predictions vs Actuals (Time Series)
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Historical
    ax3.plot(train['date'].tail(30), train['sales'].tail(30), 
             color='#2c3e50', alpha=0.5, label='Historical')
    
    # Actuals in test period
    ax3.plot(test['date'], test['sales'], color='#2c3e50', linewidth=2, 
             label='Actual', marker='o', markersize=4)
    
    # Best model prediction
    best_model = results_df.loc[results_df['mae'].idxmin(), 'model']
    ax3.plot(test['date'], all_predictions[best_model], '--', 
             color='#27ae60', linewidth=2, label=f'Best: {best_model}')
    
    # Worst model prediction
    worst_model = results_df.loc[results_df['mae'].idxmax(), 'model']
    ax3.plot(test['date'], all_predictions[worst_model], '--', 
             color='#e74c3c', linewidth=1.5, alpha=0.7, label=f'Worst: {worst_model}')
    
    ax3.axvline(train['date'].max(), color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Daily Sales ($)')
    ax3.set_title('Predictions vs Actuals', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.tick_params(axis='x', rotation=45)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative Predictions
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Actual cumulative
    ax4.plot(test['date'], test['sales'].cumsum(), color='#2c3e50', 
             linewidth=3, label='Actual')
    
    # Each model's cumulative
    for model_name, preds in all_predictions.items():
        if model_name in ['Naive Baseline', 'Sklearn Stacking']:
            ax4.plot(test['date'], np.cumsum(preds), '--', 
                    linewidth=1.5, alpha=0.7, label=model_name)
    
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Sales ($)')
    ax4.set_title('Cumulative Forecast Accuracy', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.tick_params(axis='x', rotation=45)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax4.grid(True, alpha=0.3)
    
    # 5. Daily Errors by Model
    ax5 = fig.add_subplot(2, 3, 5)
    
    best_errors = all_predictions[best_model] - test['sales'].values
    worst_errors = all_predictions[worst_model] - test['sales'].values
    
    x = range(len(test))
    width = 0.35
    ax5.bar([i - width/2 for i in x], best_errors, width, label=f'{best_model}', 
            color='#27ae60', alpha=0.7)
    ax5.bar([i + width/2 for i in x], worst_errors, width, label=f'{worst_model}', 
            color='#e74c3c', alpha=0.7)
    ax5.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_xlabel('Test Day')
    ax5.set_ylabel('Prediction Error ($)')
    ax5.set_title('Daily Prediction Errors', fontweight='bold')
    ax5.legend()
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary text
    best = results_df.loc[results_df['mae'].idxmin()]
    worst = results_df.loc[results_df['mae'].idxmax()]
    
    summary_text = f"""
    MODEL VALIDATION SUMMARY
    ════════════════════════════════════════
    
    Test Period: {test['date'].min().strftime('%Y-%m-%d')} to {test['date'].max().strftime('%Y-%m-%d')}
    Actual 30-Day Sales: ${results_df['total_actual'].iloc[0]:,.0f}
    
    🏆 BEST MODEL: {best['model']}
       MAE: ${best['mae']:,.0f}
       Total Error: {best['total_error_pct']:+.1f}%
       Predicted: ${best['total_predicted']:,.0f}
    
    ❌ WORST MODEL: {worst['model']}
       MAE: ${worst['mae']:,.0f}
       Total Error: {worst['total_error_pct']:+.1f}%
       Predicted: ${worst['total_predicted']:,.0f}
    
    RECOMMENDATIONS:
    • Use {best['model']} as primary model
    • Review/retrain {worst['model']}
    • Consider removing models with >50% error
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def generate_validation_report(results_df, test):
    """Generate validation report."""
    
    best = results_df.loc[results_df['mae'].idxmin()]
    
    report = f"""
{'='*80}
                    MODEL VALIDATION REPORT
                    Backtesting on Held-Out Test Data
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

📊 TEST CONFIGURATION
{'─'*80}
Test Period:          {test['date'].min().strftime('%Y-%m-%d')} to {test['date'].max().strftime('%Y-%m-%d')}
Test Days:            {len(test)}
Actual Total Sales:   ${test['sales'].sum():,.0f}
Actual Daily Avg:     ${test['sales'].mean():,.0f}

📈 MODEL PERFORMANCE RANKING
{'─'*80}
{'Rank':<6} {'Model':<25} {'MAE':>12} {'Total Err%':>12} {'Predicted':>15}
{'─'*80}
"""
    
    sorted_results = results_df.sort_values('mae')
    for rank, (_, row) in enumerate(sorted_results.iterrows(), 1):
        emoji = '🏆' if rank == 1 else '❌' if rank == len(sorted_results) else '  '
        report += f"{emoji}{rank:<4} {row['model']:<25} ${row['mae']:>10,.0f} {row['total_error_pct']:>+10.1f}% ${row['total_predicted']:>13,.0f}\n"
    
    report += f"""
{'─'*80}

🔍 DETAILED ANALYSIS
{'─'*80}
"""
    
    for _, row in sorted_results.iterrows():
        status = "✅ GOOD" if abs(row['total_error_pct']) < 15 else "⚠️ FAIR" if abs(row['total_error_pct']) < 30 else "❌ POOR"
        report += f"""
{row['model']}:
  MAE:              ${row['mae']:,.0f}
  RMSE:             ${row['rmse']:,.0f}
  MAPE:             {row['mape']:.1f}%
  30-Day Total:     ${row['total_predicted']:,.0f}
  Error:            {row['total_error_pct']:+.1f}%
  Status:           {status}
"""
    
    # Recommendations
    report += f"""
🎯 RECOMMENDATIONS
{'─'*80}
"""
    
    # Find models with issues
    bad_models = results_df[abs(results_df['total_error_pct']) > 30]
    good_models = results_df[abs(results_df['total_error_pct']) < 15]
    
    if len(bad_models) > 0:
        report += f"\n⚠️  Models with >30% error (consider removing from ensemble):\n"
        for _, row in bad_models.iterrows():
            report += f"    • {row['model']}: {row['total_error_pct']:+.1f}% error\n"
    
    if len(good_models) > 0:
        report += f"\n✅ Models with <15% error (reliable):\n"
        for _, row in good_models.iterrows():
            report += f"    • {row['model']}: {row['total_error_pct']:+.1f}% error\n"
    
    report += f"""
📋 SUGGESTED ENSEMBLE WEIGHTS (based on validation)
{'─'*80}
"""
    
    # Calculate suggested weights (inverse of MAE)
    inv_mae = 1 / results_df['mae']
    weights = inv_mae / inv_mae.sum()
    
    for _, row in results_df.iterrows():
        weight = weights[results_df['model'] == row['model']].values[0]
        current_weight = {'Prophet': 0.45, 'SARIMAX': 0.20, 'XGBoost': 0.10, 'Sklearn Stacking': 0.25}.get(row['model'], 0)
        report += f"  {row['model']:<20}: {weight*100:>5.1f}% (current: {current_weight*100:.0f}%)\n"
    
    report += f"""
{'='*80}
                              END OF REPORT
{'='*80}
"""
    
    return report


def main():
    """Main validation pipeline."""
    
    print("=" * 80)
    print("   MODEL VALIDATION & BACKTESTING")
    print("   Evaluating models on held-out test data")
    print("=" * 80)
    
    # Load data
    daily = load_data()
    
    # Create train/test split
    train, test = create_train_test_split(daily, test_days=30)
    
    # Store all predictions
    all_predictions = {}
    results = []
    
    # 1. Naive Baseline
    preds, actuals = evaluate_naive_baseline(train, test)
    all_predictions['Naive Baseline'] = preds
    results.append(compute_metrics(preds, actuals, 'Naive Baseline'))
    print(f"   MAE: ${results[-1]['mae']:,.0f}, Total Error: {results[-1]['total_error_pct']:+.1f}%")
    
    # 2. Prophet
    preds, actuals = evaluate_prophet(train, test)
    all_predictions['Prophet'] = preds
    results.append(compute_metrics(preds, actuals, 'Prophet'))
    print(f"   MAE: ${results[-1]['mae']:,.0f}, Total Error: {results[-1]['total_error_pct']:+.1f}%")
    
    # 3. SARIMAX
    preds, actuals = evaluate_sarimax(train, test)
    all_predictions['SARIMAX'] = preds
    results.append(compute_metrics(preds, actuals, 'SARIMAX'))
    print(f"   MAE: ${results[-1]['mae']:,.0f}, Total Error: {results[-1]['total_error_pct']:+.1f}%")
    
    # 4. XGBoost (with fix)
    preds, actuals = evaluate_xgboost(train, test, fix_leakage=True)
    all_predictions['XGBoost (fixed)'] = preds
    results.append(compute_metrics(preds, actuals, 'XGBoost (fixed)'))
    print(f"   MAE: ${results[-1]['mae']:,.0f}, Total Error: {results[-1]['total_error_pct']:+.1f}%")
    
    # 5. Sklearn Stacking
    preds, actuals = evaluate_sklearn_stacking(train, test)
    all_predictions['Sklearn Stacking'] = preds
    results.append(compute_metrics(preds, actuals, 'Sklearn Stacking'))
    print(f"   MAE: ${results[-1]['mae']:,.0f}, Total Error: {results[-1]['total_error_pct']:+.1f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate outputs
    print("\n💾 Saving outputs...")
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    fig = create_validation_visualization(results_df, all_predictions, test, train)
    fig.savefig(REPORTS_PATH / "model_validation.png", dpi=150, bbox_inches='tight')
    print(f"   Plot: {REPORTS_PATH / 'model_validation.png'}")
    
    # Results CSV
    results_df.to_csv(REPORTS_PATH / "validation_results.csv", index=False)
    print(f"   Results: {REPORTS_PATH / 'validation_results.csv'}")
    
    # Report
    report = generate_validation_report(results_df, test)
    with open(REPORTS_PATH / "validation_report.txt", 'w') as f:
        f.write(report)
    print(f"   Report: {REPORTS_PATH / 'validation_report.txt'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("   📊 VALIDATION SUMMARY")
    print("=" * 80)
    
    best = results_df.loc[results_df['mae'].idxmin()]
    worst = results_df.loc[results_df['mae'].idxmax()]
    
    print(f"\n   🏆 BEST MODEL: {best['model']}")
    print(f"      MAE: ${best['mae']:,.0f}")
    print(f"      30-Day Error: {best['total_error_pct']:+.1f}%")
    
    print(f"\n   ❌ WORST MODEL: {worst['model']}")
    print(f"      MAE: ${worst['mae']:,.0f}")
    print(f"      30-Day Error: {worst['total_error_pct']:+.1f}%")
    
    print(f"\n   Actual 30-Day Sales: ${test['sales'].sum():,.0f}")
    print("=" * 80)
    
    print("\n✅ Validation complete!")
    
    return results_df, all_predictions, test


if __name__ == "__main__":
    results_df, all_predictions, test = main()

