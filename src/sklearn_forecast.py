"""
Scikit-Learn Sales Forecasting
==============================
Multiple sklearn algorithms with stacking ensemble for robust predictions.

Models included:
- Ridge Regression (regularized linear)
- Lasso Regression (feature selection)
- Random Forest (bagged trees)
- Gradient Boosting (sklearn's implementation)
- MLPRegressor (neural network)
- Stacking Ensemble (meta-learner combining all)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    StackingRegressor,
    VotingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_sales_history.json"
REPORTS_PATH = PROJECT_ROOT / "reports" / "forecasts" / "sklearn"


def load_and_prepare_data():
    """Load combined sales data and prepare for sklearn models."""
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
    print(f"   Daily avg: ${daily['sales'].mean():,.0f}")
    
    return daily


def create_features(df, exclude_leaky=True):
    """
    Create comprehensive time-series features for sklearn models.
    
    Args:
        df: DataFrame with date and sales columns
        exclude_leaky: If True, exclude features that cause data leakage 
                       (invoice_count, customer_count) since we won't have 
                       these for future predictions
    """
    df = df.copy()
    
    # === Date/Calendar Features ===
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding for day of week and month (sin/cos transform)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # === Lag Features (Sales) ===
    for lag in [1, 2, 3, 5, 7, 14, 21, 28]:
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    
    # === Rolling Statistics (Sales) ===
    for window in [7, 14, 21, 28]:
        df[f'rolling_mean_{window}'] = df['sales'].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['sales'].shift(1).rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['sales'].shift(1).rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['sales'].shift(1).rolling(window=window).max()
        df[f'rolling_median_{window}'] = df['sales'].shift(1).rolling(window=window).median()
    
    # === Expanding Statistics ===
    df['expanding_mean'] = df['sales'].shift(1).expanding().mean()
    df['expanding_std'] = df['sales'].shift(1).expanding().std()
    
    # === Same-Period Comparisons ===
    df['same_day_last_week'] = df['sales'].shift(7)
    df['same_day_2_weeks_ago'] = df['sales'].shift(14)
    df['same_day_3_weeks_ago'] = df['sales'].shift(21)
    df['same_day_4_weeks_ago'] = df['sales'].shift(28)
    
    # Week-over-week change
    df['wow_change'] = df['sales'].shift(1) - df['sales'].shift(8)
    df['wow_pct_change'] = (df['sales'].shift(1) / df['sales'].shift(8).replace(0, np.nan)) - 1
    
    # === Trend Features ===
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    
    # === Volatility Features ===
    df['rolling_range_7'] = df['rolling_max_7'] - df['rolling_min_7']
    df['rolling_range_14'] = df['rolling_max_14'] - df['rolling_min_14']
    
    # Coefficient of variation
    df['rolling_cv_7'] = df['rolling_std_7'] / df['rolling_mean_7'].replace(0, np.nan)
    df['rolling_cv_14'] = df['rolling_std_14'] / df['rolling_mean_14'].replace(0, np.nan)
    
    # === Exclude leaky features ===
    # invoice_count and customer_count are highly correlated with sales
    # but we don't have them for future predictions
    if exclude_leaky:
        leaky_cols = ['invoice_count', 'customer_count']
        for col in leaky_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
    
    return df


def create_sklearn_models():
    """Create dictionary of sklearn models to evaluate."""
    
    models = {
        'Ridge': Pipeline([
            ('scaler', RobustScaler()),
            ('model', Ridge(alpha=1.0, random_state=42))
        ]),
        
        'Lasso': Pipeline([
            ('scaler', RobustScaler()),
            ('model', Lasso(alpha=100, random_state=42, max_iter=5000))
        ]),
        
        'ElasticNet': Pipeline([
            ('scaler', RobustScaler()),
            ('model', ElasticNet(alpha=100, l1_ratio=0.5, random_state=42, max_iter=5000))
        ]),
        
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            random_state=42
        ),
        
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                random_state=42
            ))
        ]),
        
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=1000, epsilon=0.1))
        ])
    }
    
    return models


def create_stacking_ensemble(models):
    """Create stacking ensemble with meta-learner."""
    
    # Base estimators (use a subset for efficiency - tree-based only for stability)
    estimators = [
        ('rf', models['RandomForest']),
        ('gbr', models['GradientBoosting']),
    ]
    
    # Stacking with Ridge as meta-learner (use simple int cv for stability)
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,  # Simple int instead of TimeSeriesSplit for stability
        passthrough=False,
        n_jobs=-1
    )
    
    return stacking


def create_voting_ensemble(models):
    """Create voting ensemble (weighted average)."""
    
    estimators = [
        ('ridge', models['Ridge']),
        ('rf', models['RandomForest']),
        ('gbr', models['GradientBoosting']),
        ('mlp', models['MLP'])
    ]
    
    # Voting regressor (average predictions)
    voting = VotingRegressor(
        estimators=estimators,
        n_jobs=-1
    )
    
    return voting


def evaluate_models(X_train, y_train, models):
    """Evaluate all models using time series cross-validation."""
    
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    
    print("\n📊 Model Evaluation (TimeSeriesSplit CV)")
    print("─" * 60)
    print(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'R²':>10}")
    print("─" * 60)
    
    for name, model in models.items():
        try:
            # Cross-validation for MAE
            mae_scores = -cross_val_score(model, X_train, y_train, cv=tscv, 
                                          scoring='neg_mean_absolute_error', n_jobs=-1)
            
            # Cross-validation for R²
            r2_scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                        scoring='r2', n_jobs=-1)
            
            # RMSE from MSE
            mse_scores = -cross_val_score(model, X_train, y_train, cv=tscv,
                                          scoring='neg_mean_squared_error', n_jobs=-1)
            rmse_scores = np.sqrt(mse_scores)
            
            results[name] = {
                'mae': mae_scores.mean(),
                'mae_std': mae_scores.std(),
                'rmse': rmse_scores.mean(),
                'rmse_std': rmse_scores.std(),
                'r2': r2_scores.mean(),
                'r2_std': r2_scores.std()
            }
            
            print(f"{name:<20} ${mae_scores.mean():>10,.0f} ${rmse_scores.mean():>10,.0f} {r2_scores.mean():>10.3f}")
            
        except Exception as e:
            print(f"{name:<20} Error: {str(e)[:30]}")
            results[name] = {'mae': np.inf, 'rmse': np.inf, 'r2': -np.inf}
    
    print("─" * 60)
    
    return results


def train_and_forecast(df, models, forecast_days=30):
    """Train models and generate forecasts."""
    
    # Create features (exclude leaky features like invoice_count)
    df_features = create_features(df, exclude_leaky=True)
    
    # Drop rows with NaN (due to lag features)
    df_clean = df_features.dropna()
    
    # Define features and target
    feature_cols = [col for col in df_clean.columns if col not in ['date', 'sales']]
    X = df_clean[feature_cols].values
    y = df_clean['sales'].values
    
    print(f"\n📊 Feature Engineering Complete")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Training samples: {len(X)}")
    
    # Evaluate models
    cv_results = evaluate_models(X, y, models)
    
    # Create ensembles
    print("\n🔧 Creating Ensemble Models...")
    stacking = create_stacking_ensemble(models)
    voting = create_voting_ensemble(models)
    
    # Evaluate ensembles
    ensemble_models = {'Stacking': stacking, 'Voting': voting}
    ensemble_results = evaluate_models(X, y, ensemble_models)
    cv_results.update(ensemble_results)
    
    # Select best individual model and best ensemble
    best_individual = min([(k, v) for k, v in cv_results.items() 
                           if k not in ['Stacking', 'Voting']], 
                          key=lambda x: x[1]['mae'])
    best_ensemble = min([(k, v) for k, v in cv_results.items() 
                         if k in ['Stacking', 'Voting']], 
                        key=lambda x: x[1]['mae'])
    
    print(f"\n🏆 Best Individual Model: {best_individual[0]} (MAE: ${best_individual[1]['mae']:,.0f})")
    print(f"🏆 Best Ensemble: {best_ensemble[0]} (MAE: ${best_ensemble[1]['mae']:,.0f})")
    
    # Train all models on full data
    print("\n🎓 Training models on full dataset...")
    trained_models = {}
    for name, model in {**models, 'Stacking': stacking, 'Voting': voting}.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
        except Exception as e:
            print(f"   Warning: {name} failed to train: {e}")
    
    # Generate forecasts
    print(f"\n🔮 Generating {forecast_days}-day forecasts...")
    
    last_date = df['date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=forecast_days, freq='D')
    
    # Store forecasts from each model
    all_forecasts = {name: [] for name in trained_models.keys()}
    
    # Iteratively forecast
    forecast_df = df.copy()
    
    for future_date in forecast_dates:
        # Add new row (don't need invoice_count/customer_count since they're excluded)
        new_row = pd.DataFrame({
            'date': [future_date], 
            'sales': [np.nan]
        })
        # Add other columns if they exist in forecast_df
        for col in ['invoice_count', 'customer_count']:
            if col in forecast_df.columns:
                new_row[col] = 0
        
        forecast_df = pd.concat([forecast_df, new_row], ignore_index=True)
        
        # Recreate features (exclude leaky features)
        forecast_df = create_features(forecast_df, exclude_leaky=True)
        
        # Get the last row features
        last_row = forecast_df.iloc[-1:][feature_cols].values
        
        # Predict with each model
        predictions = {}
        for name, model in trained_models.items():
            try:
                pred = model.predict(last_row)[0]
                pred = max(0, pred)  # Floor at 0
                predictions[name] = pred
                all_forecasts[name].append(pred)
            except Exception as e:
                predictions[name] = 0
                all_forecasts[name].append(0)
        
        # Use stacking prediction for the next iteration
        forecast_df.loc[forecast_df['date'] == future_date, 'sales'] = predictions.get('Stacking', 
                                                                                         predictions.get('Voting', 0))
    
    # Create forecast results DataFrame
    forecast_results = pd.DataFrame({
        'date': forecast_dates,
        **{name: preds for name, preds in all_forecasts.items()}
    })
    
    # Add ensemble average
    model_cols = [col for col in forecast_results.columns if col != 'date']
    forecast_results['ensemble_avg'] = forecast_results[model_cols].mean(axis=1)
    
    # Calculate feature importance from Random Forest
    feature_importance = None
    if 'RandomForest' in trained_models:
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': trained_models['RandomForest'].feature_importances_
        }).sort_values('importance', ascending=False)
    
    return trained_models, forecast_results, cv_results, df_clean, feature_cols, feature_importance


def create_visualization(df_clean, forecast_results, cv_results, feature_importance):
    """Create comprehensive sklearn forecast visualization."""
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Scikit-Learn Sales Forecasting', fontsize=16, fontweight='bold', y=1.02)
    
    # Color palette
    colors = {
        'Ridge': '#3498db',
        'Lasso': '#9b59b6',
        'ElasticNet': '#1abc9c',
        'RandomForest': '#27ae60',
        'GradientBoosting': '#e74c3c',
        'MLP': '#f39c12',
        'SVR': '#e91e63',
        'Stacking': '#2c3e50',
        'Voting': '#8e44ad',
        'Historical': '#7f8c8d'
    }
    
    # 1. Model Comparison (CV Performance)
    ax1 = fig.add_subplot(2, 3, 1)
    
    model_names = list(cv_results.keys())
    maes = [cv_results[m]['mae'] for m in model_names]
    mae_stds = [cv_results[m].get('mae_std', 0) for m in model_names]
    
    bar_colors = [colors.get(m, '#95a5a6') for m in model_names]
    bars = ax1.barh(range(len(model_names)), maes, xerr=mae_stds, 
                    color=bar_colors, alpha=0.8, capsize=3)
    
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names)
    ax1.set_xlabel('Mean Absolute Error ($)')
    ax1.set_title('Model CV Performance (MAE)', fontweight='bold')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Highlight best
    best_idx = np.argmin(maes)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # 2. R² Comparison
    ax2 = fig.add_subplot(2, 3, 2)
    
    r2s = [cv_results[m].get('r2', 0) for m in model_names]
    r2_colors = ['#27ae60' if r > 0.5 else '#e74c3c' if r < 0 else '#f39c12' for r in r2s]
    
    ax2.barh(range(len(model_names)), r2s, color=r2_colors, alpha=0.8)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel('R² Score')
    ax2.set_title('Model CV Performance (R²)', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Historical + All Forecasts
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Historical (last 60 days)
    recent_hist = df_clean.tail(60)
    ax3.plot(recent_hist['date'], recent_hist['sales'], color=colors['Historical'], 
             alpha=0.5, linewidth=1, label='Historical')
    
    # Plot each model's forecast
    forecast_cols = [col for col in forecast_results.columns if col not in ['date', 'ensemble_avg']]
    for col in forecast_cols:
        ax3.plot(forecast_results['date'], forecast_results[col], 
                 '--', color=colors.get(col, '#95a5a6'), alpha=0.5, linewidth=1)
    
    # Ensemble average (bold)
    ax3.plot(forecast_results['date'], forecast_results['ensemble_avg'],
             color='#2c3e50', linewidth=3, label='Ensemble Avg', marker='o', markersize=3)
    
    # Stacking (bold, different color)
    if 'Stacking' in forecast_results.columns:
        ax3.plot(forecast_results['date'], forecast_results['Stacking'],
                 color=colors['Stacking'], linewidth=2.5, linestyle='--', label='Stacking')
    
    ax3.axvline(df_clean['date'].max(), color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Daily Sales ($)')
    ax3.set_title('30-Day Forecast Comparison', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.tick_params(axis='x', rotation=45)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature Importance
    ax4 = fig.add_subplot(2, 3, 4)
    
    if feature_importance is not None:
        top_features = feature_importance.head(15)
        feat_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        
        ax4.barh(range(len(top_features)), top_features['importance'], color=feat_colors)
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels(top_features['feature'], fontsize=8)
        ax4.set_xlabel('Importance')
        ax4.set_title('Top 15 Features (Random Forest)', fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
    else:
        ax4.text(0.5, 0.5, 'Feature importance\nnot available', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Feature Importance', fontweight='bold')
    
    # 5. Model Spread / Uncertainty
    ax5 = fig.add_subplot(2, 3, 5)
    
    forecast_cols = [col for col in forecast_results.columns if col not in ['date', 'ensemble_avg']]
    model_predictions = forecast_results[forecast_cols]
    
    # Calculate spread
    pred_min = model_predictions.min(axis=1)
    pred_max = model_predictions.max(axis=1)
    pred_mean = model_predictions.mean(axis=1)
    
    ax5.fill_between(forecast_results['date'], pred_min, pred_max, 
                     alpha=0.3, color='#3498db', label='Model Range')
    ax5.plot(forecast_results['date'], pred_mean, 
             color='#2c3e50', linewidth=2, label='Mean')
    
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Predicted Sales ($)')
    ax5.set_title('Model Uncertainty (Spread)', fontweight='bold')
    ax5.legend()
    ax5.tick_params(axis='x', rotation=45)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative Forecast by Model
    ax6 = fig.add_subplot(2, 3, 6)
    
    for col in ['RandomForest', 'GradientBoosting', 'Stacking', 'Voting']:
        if col in forecast_results.columns:
            cumsum = forecast_results[col].cumsum()
            ax6.plot(forecast_results['date'], cumsum, 
                    color=colors.get(col, '#95a5a6'), linewidth=2, label=col)
    
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Cumulative Sales ($)')
    ax6.set_title('Cumulative 30-Day Forecast', fontweight='bold')
    ax6.legend(loc='upper left')
    ax6.tick_params(axis='x', rotation=45)
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_report(forecast_results, cv_results, feature_importance, df_clean):
    """Generate comprehensive sklearn forecast report."""
    
    # Find best models
    sorted_by_mae = sorted(cv_results.items(), key=lambda x: x[1]['mae'])
    best_model = sorted_by_mae[0]
    
    forecast_cols = [col for col in forecast_results.columns if col not in ['date', 'ensemble_avg']]
    
    report = f"""
{'='*80}
                    SCIKIT-LEARN SALES FORECAST REPORT
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

📊 MODEL COMPARISON (Cross-Validation)
{'─'*80}
{'Model':<20} {'MAE':>12} {'RMSE':>12} {'R²':>10}
{'─'*80}
"""
    
    for name, metrics in sorted_by_mae:
        mae_str = f"${metrics['mae']:,.0f}" if metrics['mae'] != np.inf else "Error"
        rmse_str = f"${metrics['rmse']:,.0f}" if metrics['rmse'] != np.inf else "Error"
        r2_str = f"{metrics['r2']:.3f}" if metrics['r2'] != -np.inf else "Error"
        report += f"{name:<20} {mae_str:>12} {rmse_str:>12} {r2_str:>10}\n"
    
    report += f"""{'─'*80}

🏆 BEST PERFORMING MODEL
{'─'*80}
Model:              {best_model[0]}
CV MAE:             ${best_model[1]['mae']:,.0f}
CV RMSE:            ${best_model[1]['rmse']:,.0f}
CV R²:              {best_model[1]['r2']:.3f}

📈 30-DAY FORECAST SUMMARY
{'─'*80}
Forecast Period:    {forecast_results['date'].min().strftime('%Y-%m-%d')} to {forecast_results['date'].max().strftime('%Y-%m-%d')}

{'Model':<20} {'30-Day Total':>15} {'Daily Avg':>12}
{'─'*80}
"""
    
    for col in forecast_cols:
        total = forecast_results[col].sum()
        avg = forecast_results[col].mean()
        report += f"{col:<20} ${total:>14,.0f} ${avg:>10,.0f}\n"
    
    report += f"{'─'*80}\n"
    report += f"{'Ensemble Average':<20} ${forecast_results['ensemble_avg'].sum():>14,.0f} ${forecast_results['ensemble_avg'].mean():>10,.0f}\n"
    
    # Model agreement / uncertainty
    model_predictions = forecast_results[forecast_cols]
    pred_std = model_predictions.std(axis=1).mean()
    pred_cv = (model_predictions.std(axis=1) / model_predictions.mean(axis=1)).mean()
    
    report += f"""
📊 MODEL AGREEMENT / UNCERTAINTY
{'─'*80}
Average Daily Std Dev:          ${pred_std:,.0f}
Average Coef. of Variation:     {pred_cv:.1%}
Model Consensus:                {'Strong' if pred_cv < 0.3 else 'Moderate' if pred_cv < 0.5 else 'Weak'}

"""
    
    # Feature importance
    if feature_importance is not None:
        report += f"""🔍 TOP 10 FEATURES (Random Forest Importance)
{'─'*80}
"""
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            report += f"  {i:2}. {row['feature']:<30} {row['importance']:.4f}\n"
    
    # Daily breakdown
    report += f"""
📅 DAILY FORECAST (Stacking Ensemble)
{'─'*80}
{'Date':<12} {'Stacking':>12} {'RF':>10} {'GBR':>10} {'MLP':>10} {'Ensemble':>10}
{'─'*80}
"""
    
    for _, row in forecast_results.iterrows():
        stacking = f"${row.get('Stacking', 0):,.0f}"
        rf = f"${row.get('RandomForest', 0):,.0f}"
        gbr = f"${row.get('GradientBoosting', 0):,.0f}"
        mlp = f"${row.get('MLP', 0):,.0f}"
        ensemble = f"${row['ensemble_avg']:,.0f}"
        report += f"{row['date'].strftime('%Y-%m-%d'):<12} {stacking:>12} {rf:>10} {gbr:>10} {mlp:>10} {ensemble:>10}\n"
    
    # Historical comparison
    hist_daily_avg = df_clean['sales'].mean()
    stacking_total = forecast_results.get('Stacking', forecast_results['ensemble_avg']).sum()
    
    report += f"""
📊 COMPARISON TO HISTORICAL
{'─'*80}
Historical Daily Avg:           ${hist_daily_avg:,.0f}
Stacking 30-Day Forecast:       ${stacking_total:,.0f}
Historical 30-Day Est:          ${hist_daily_avg * 30:,.0f}
vs Historical:                  {(stacking_total / (hist_daily_avg * 30) - 1) * 100:+.1f}%

{'='*80}
                              END OF REPORT
{'='*80}
"""
    
    return report


def main():
    """Main sklearn forecasting pipeline."""
    
    print("=" * 80)
    print("   SCIKIT-LEARN SALES FORECASTING")
    print("   Multiple Models + Stacking Ensemble")
    print("=" * 80)
    
    # Load data
    daily = load_and_prepare_data()
    
    # Create models
    print("\n🔧 Creating sklearn models...")
    models = create_sklearn_models()
    print(f"   Models: {', '.join(models.keys())}")
    
    # Train and forecast
    trained_models, forecast_results, cv_results, df_clean, feature_cols, feature_importance = \
        train_and_forecast(daily, models, forecast_days=30)
    
    # Generate outputs
    print("\n💾 Saving outputs...")
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    fig = create_visualization(df_clean, forecast_results, cv_results, feature_importance)
    fig.savefig(REPORTS_PATH / "sklearn_forecast.png", dpi=150, bbox_inches='tight')
    print(f"   Plot: {REPORTS_PATH / 'sklearn_forecast.png'}")
    
    # Forecast CSV
    forecast_results.to_csv(REPORTS_PATH / "sklearn_forecast.csv", index=False)
    print(f"   Forecast data: {REPORTS_PATH / 'sklearn_forecast.csv'}")
    
    # Model comparison CSV
    cv_df = pd.DataFrame(cv_results).T
    cv_df.to_csv(REPORTS_PATH / "sklearn_model_comparison.csv")
    print(f"   Model comparison: {REPORTS_PATH / 'sklearn_model_comparison.csv'}")
    
    # Feature importance CSV
    if feature_importance is not None:
        feature_importance.to_csv(REPORTS_PATH / "sklearn_feature_importance.csv", index=False)
        print(f"   Feature importance: {REPORTS_PATH / 'sklearn_feature_importance.csv'}")
    
    # Report
    report = generate_report(forecast_results, cv_results, feature_importance, df_clean)
    with open(REPORTS_PATH / "sklearn_forecast_report.txt", 'w') as f:
        f.write(report)
    print(f"   Report: {REPORTS_PATH / 'sklearn_forecast_report.txt'}")
    
    # Print summary
    best_model = min(cv_results.items(), key=lambda x: x[1]['mae'])
    stacking_total = forecast_results.get('Stacking', forecast_results['ensemble_avg']).sum()
    
    print("\n" + "=" * 80)
    print("   📈 SKLEARN 30-DAY FORECAST SUMMARY")
    print("=" * 80)
    print(f"   Best Model:      {best_model[0]} (MAE: ${best_model[1]['mae']:,.0f})")
    print(f"   Period:          {forecast_results['date'].min().strftime('%Y-%m-%d')} to {forecast_results['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Stacking Total:  ${stacking_total:>12,.0f}")
    print(f"   Daily Average:   ${stacking_total / 30:>12,.0f}")
    print("=" * 80)
    
    print("\n✅ Scikit-learn forecasting complete!")
    
    return trained_models, forecast_results, cv_results


if __name__ == "__main__":
    trained_models, forecast_results, cv_results = main()

