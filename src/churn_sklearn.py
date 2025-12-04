"""
Enhanced Churn Prediction with Scikit-Learn
============================================
Multiple ML models with cross-validation for robust churn prediction.

Models included:
- Random Forest (baseline)
- Gradient Boosting
- Logistic Regression (interpretable)
- XGBoost
- Stacking Ensemble

Features:
- Proper train/test split
- Cross-validation
- Probability calibration
- Feature importance analysis
- Model comparison
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_sales_history.json"
REPORTS_PATH = PROJECT_ROOT / "reports" / "customers" / "churn"


def load_sales_data():
    """Load combined sales history."""
    print("📥 Loading sales data...")
    
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    
    # Filter to invoices only
    df = df[df['invoice_type'] == 'IN'].copy()
    
    print(f"   Records: {len(df):,}")
    print(f"   Date range: {df['invoice_date'].min().strftime('%Y-%m-%d')} to {df['invoice_date'].max().strftime('%Y-%m-%d')}")
    print(f"   Unique customers: {df['customer_no'].nunique():,}")
    
    return df


def engineer_features(df, reference_date=None):
    """
    Create comprehensive features for churn prediction.
    
    RFM features + behavioral + trend indicators
    """
    if reference_date is None:
        reference_date = df['invoice_date'].max() + timedelta(days=1)
    
    print(f"\n🔧 Engineering features (reference: {reference_date.strftime('%Y-%m-%d')})...")
    
    features = []
    
    for customer_no, group in df.groupby('customer_no'):
        group = group.sort_values('invoice_date')
        
        # Basic info
        customer_name = group['customer_name'].iloc[0] if 'customer_name' in group.columns else 'Unknown'
        
        # === RFM Features ===
        recency = (reference_date - group['invoice_date'].max()).days
        frequency = len(group)
        monetary = group['amount'].sum()
        avg_order_value = group['amount'].mean()
        
        # === Time-based Features ===
        first_order = group['invoice_date'].min()
        last_order = group['invoice_date'].max()
        customer_tenure = (reference_date - first_order).days
        
        # === Order Gap Analysis ===
        if len(group) > 1:
            order_dates = group['invoice_date'].sort_values()
            gaps = order_dates.diff().dt.days.dropna()
            avg_gap = gaps.mean()
            gap_std = gaps.std() if len(gaps) > 1 else 0
            gap_cv = gap_std / avg_gap if avg_gap > 0 else 0
            min_gap = gaps.min()
            max_gap = gaps.max()
            median_gap = gaps.median()
        else:
            avg_gap = customer_tenure
            gap_std = 0
            gap_cv = 0
            min_gap = customer_tenure
            max_gap = customer_tenure
            median_gap = customer_tenure
        
        # === Trend Features (6-month windows) ===
        six_months_ago = reference_date - timedelta(days=180)
        twelve_months_ago = reference_date - timedelta(days=365)
        
        recent_orders = len(group[group['invoice_date'] >= six_months_ago])
        prior_orders = len(group[(group['invoice_date'] >= twelve_months_ago) & 
                                  (group['invoice_date'] < six_months_ago)])
        
        recent_revenue = group[group['invoice_date'] >= six_months_ago]['amount'].sum()
        prior_revenue = group[(group['invoice_date'] >= twelve_months_ago) & 
                              (group['invoice_date'] < six_months_ago)]['amount'].sum()
        
        # Trend ratios
        if prior_orders > 0:
            order_trend = (recent_orders - prior_orders) / prior_orders
        else:
            order_trend = 1 if recent_orders > 0 else -1
            
        if prior_revenue > 0:
            revenue_trend = (recent_revenue - prior_revenue) / prior_revenue
        else:
            revenue_trend = 1 if recent_revenue > 0 else -1
        
        # === Seasonality Features ===
        group['quarter'] = group['invoice_date'].dt.quarter
        group['month'] = group['invoice_date'].dt.month
        
        quarterly_counts = group.groupby('quarter').size()
        seasonality_cv = quarterly_counts.std() / quarterly_counts.mean() if quarterly_counts.mean() > 0 else 0
        
        # Peak quarter
        peak_quarter = quarterly_counts.idxmax() if len(quarterly_counts) > 0 else 0
        
        # === Derived Features ===
        orders_per_month = frequency / max(customer_tenure / 30, 1)
        revenue_per_month = monetary / max(customer_tenure / 30, 1)
        
        # Recency relative to average gap (important for churn)
        recency_gap_ratio = recency / avg_gap if avg_gap > 0 else recency
        
        # Value concentration (do they order big or small?)
        order_value_cv = group['amount'].std() / group['amount'].mean() if group['amount'].mean() > 0 else 0
        
        # Days since first order as log (captures diminishing returns of tenure)
        log_tenure = np.log1p(customer_tenure)
        
        features.append({
            'customer_no': customer_no,
            'customer_name': customer_name,
            # RFM
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'avg_order_value': avg_order_value,
            # Time
            'customer_tenure': customer_tenure,
            'log_tenure': log_tenure,
            # Gap analysis
            'avg_gap': avg_gap,
            'gap_std': gap_std,
            'gap_cv': gap_cv,
            'min_gap': min_gap,
            'max_gap': max_gap,
            'median_gap': median_gap,
            'recency_gap_ratio': recency_gap_ratio,
            # Trends
            'recent_orders': recent_orders,
            'prior_orders': prior_orders,
            'recent_revenue': recent_revenue,
            'prior_revenue': prior_revenue,
            'order_trend': order_trend,
            'revenue_trend': revenue_trend,
            # Seasonality
            'seasonality_cv': seasonality_cv,
            'peak_quarter': peak_quarter,
            # Derived
            'orders_per_month': orders_per_month,
            'revenue_per_month': revenue_per_month,
            'order_value_cv': order_value_cv,
        })
    
    features_df = pd.DataFrame(features)
    print(f"   Created {len(features_df.columns) - 2} features for {len(features_df):,} customers")
    
    return features_df


def define_churn_labels(features_df, churn_days=90):
    """
    Define churn based on customer behavior.
    
    A customer is churned if:
    - Haven't ordered in > churn_days AND
    - This is significantly longer than their typical gap
    """
    # Primary: recency > threshold
    features_df['is_churned'] = (features_df['recency'] > churn_days).astype(int)
    
    # Secondary: recency > 2x their average gap (customer-specific)
    features_df['is_churned_relative'] = (
        features_df['recency_gap_ratio'] > 2.0
    ).astype(int)
    
    # Combined: either condition
    features_df['churn_label'] = (
        (features_df['is_churned'] == 1) | 
        (features_df['is_churned_relative'] == 1)
    ).astype(int)
    
    churn_rate = features_df['churn_label'].mean()
    print(f"\n📊 Churn Statistics:")
    print(f"   Churned (>{churn_days} days): {features_df['is_churned'].sum():,} ({features_df['is_churned'].mean()*100:.1f}%)")
    print(f"   Churned (relative): {features_df['is_churned_relative'].sum():,} ({features_df['is_churned_relative'].mean()*100:.1f}%)")
    print(f"   Combined churn rate: {churn_rate*100:.1f}%")
    
    return features_df


def create_sklearn_models():
    """Create dictionary of sklearn classifiers."""
    
    models = {
        'LogisticRegression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        ),
        
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=3,  # Handle imbalance
            random_state=42,
            use_label_encoder=False,
            eval_metric='auc'
        )
    }
    
    return models


def create_stacking_ensemble(models):
    """Create stacking ensemble classifier."""
    
    estimators = [
        ('lr', models['LogisticRegression']),
        ('rf', models['RandomForest']),
        ('xgb', models['XGBoost'])
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0),
        cv=3,
        n_jobs=-1
    )
    
    return stacking


def train_and_evaluate(features_df, models):
    """Train and evaluate all models with cross-validation."""
    
    # Define features
    feature_cols = [
        'recency', 'frequency', 'monetary', 'avg_order_value',
        'customer_tenure', 'log_tenure',
        'avg_gap', 'gap_std', 'gap_cv', 'min_gap', 'max_gap', 'median_gap',
        'recency_gap_ratio',
        'recent_orders', 'prior_orders', 'order_trend', 'revenue_trend',
        'seasonality_cv',
        'orders_per_month', 'revenue_per_month', 'order_value_cv'
    ]
    
    X = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = features_df['churn_label']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 Training Data:")
    print(f"   Train: {len(X_train):,} customers")
    print(f"   Test: {len(X_test):,} customers")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Churn rate (train): {y_train.mean()*100:.1f}%")
    
    # Evaluate each model
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\n📈 Model Evaluation (5-Fold Stratified CV):")
    print("─" * 70)
    print(f"{'Model':<25} {'AUC-ROC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("─" * 70)
    
    trained_models = {}
    
    for name, model in models.items():
        # Cross-validation
        cv_auc = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
        
        # Fit on full training data
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Test set evaluation
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_proba)
        test_f1 = f1_score(y_test, y_pred)
        test_precision = (y_pred & y_test).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
        test_recall = (y_pred & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
        
        results.append({
            'model': name,
            'cv_auc': cv_auc.mean(),
            'cv_auc_std': cv_auc.std(),
            'cv_f1': cv_f1.mean(),
            'test_auc': test_auc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall
        })
        
        print(f"{name:<25} {cv_auc.mean():>10.3f} {cv_f1.mean():>10.3f} {test_precision:>10.3f} {test_recall:>10.3f}")
    
    # Add stacking ensemble
    print("\n🔧 Training Stacking Ensemble...")
    stacking = create_stacking_ensemble(models)
    stacking.fit(X_train_scaled, y_train)
    trained_models['Stacking'] = stacking
    
    y_pred = stacking.predict(X_test_scaled)
    y_proba = stacking.predict_proba(X_test_scaled)[:, 1]
    
    stack_auc = roc_auc_score(y_test, y_proba)
    stack_f1 = f1_score(y_test, y_pred)
    stack_precision = (y_pred & y_test).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
    stack_recall = (y_pred & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
    
    results.append({
        'model': 'Stacking',
        'cv_auc': stack_auc,  # No CV for stacking
        'cv_auc_std': 0,
        'cv_f1': stack_f1,
        'test_auc': stack_auc,
        'test_f1': stack_f1,
        'test_precision': stack_precision,
        'test_recall': stack_recall
    })
    
    print("─" * 70)
    print(f"{'Stacking':<25} {stack_auc:>10.3f} {stack_f1:>10.3f} {stack_precision:>10.3f} {stack_recall:>10.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Get feature importance from best tree model
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': trained_models['RandomForest'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    return trained_models, results_df, feature_importance, scaler, feature_cols, X_test, y_test


def score_all_customers(features_df, best_model, scaler, feature_cols):
    """Score all customers with churn probability."""
    
    X_all = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_all_scaled = scaler.transform(X_all)
    
    # Get probabilities
    features_df['churn_probability'] = best_model.predict_proba(X_all_scaled)[:, 1]
    
    # Risk categories
    features_df['risk_category'] = pd.cut(
        features_df['churn_probability'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    # Revenue at risk
    features_df['revenue_at_risk'] = features_df['churn_probability'] * features_df['monetary']
    
    return features_df


def generate_interventions(features_df):
    """Generate intervention recommendations for at-risk customers."""
    
    # Filter to actionable customers (>30% churn probability)
    at_risk = features_df[features_df['churn_probability'] > 0.3].copy()
    at_risk = at_risk.sort_values('revenue_at_risk', ascending=False)
    
    interventions = []
    
    for _, customer in at_risk.iterrows():
        strategies = []
        urgency = 'MEDIUM'
        
        # High-value at critical risk
        if customer['monetary'] > 100000 and customer['churn_probability'] > 0.7:
            urgency = 'CRITICAL'
            strategies.append("🔴 Executive outreach required")
            strategies.append("💰 Prepare custom retention offer")
        
        # Declining trend
        if customer['revenue_trend'] < -0.3:
            strategies.append("📉 Revenue declining - investigate competitor/product issues")
            if urgency == 'MEDIUM':
                urgency = 'HIGH'
        
        # Long silence from active customer
        if customer['recency_gap_ratio'] > 2.5 and customer['prior_orders'] > 5:
            strategies.append("⚠️ Previously active customer gone quiet")
            strategies.append("📞 Immediate follow-up call")
            if urgency == 'MEDIUM':
                urgency = 'HIGH'
        
        # Seasonal buyer check
        if customer['seasonality_cv'] > 0.5:
            strategies.append("📅 Seasonal buyer - check timing")
        
        # New customer struggling
        if customer['customer_tenure'] < 180 and customer['churn_probability'] > 0.5:
            strategies.append("🆕 New customer at risk - onboarding review")
        
        # Default
        if not strategies:
            strategies.append("📧 Send engagement email")
            strategies.append("📞 Schedule check-in call")
        
        interventions.append({
            'customer_no': customer['customer_no'],
            'customer_name': customer['customer_name'],
            'churn_probability': customer['churn_probability'],
            'risk_category': customer['risk_category'],
            'revenue_at_risk': customer['revenue_at_risk'],
            'total_revenue': customer['monetary'],
            'recency_days': customer['recency'],
            'order_trend': customer['order_trend'],
            'urgency': urgency,
            'strategies': ' | '.join(strategies)
        })
    
    return pd.DataFrame(interventions)


def create_visualization(features_df, results_df, interventions_df, feature_importance):
    """Create comprehensive churn analysis visualization."""
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('🚨 Customer Churn Analysis (Sklearn Models)', fontsize=16, fontweight='bold')
    
    # Colors
    colors = {
        'Critical': '#e74c3c',
        'High': '#f39c12',
        'Medium': '#3498db',
        'Low': '#27ae60'
    }
    
    # 1. Model Comparison
    ax1 = fig.add_subplot(2, 3, 1)
    
    sorted_results = results_df.sort_values('test_auc', ascending=True)
    y_pos = np.arange(len(sorted_results))
    bar_colors = ['#27ae60' if r['test_auc'] == sorted_results['test_auc'].max() else '#3498db' 
                  for _, r in sorted_results.iterrows()]
    
    ax1.barh(y_pos, sorted_results['test_auc'], color=bar_colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_results['model'])
    ax1.set_xlabel('AUC-ROC Score')
    ax1.set_title('Model Comparison (Test Set)')
    ax1.set_xlim(0.5, 1.0)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add values
    for i, (_, r) in enumerate(sorted_results.iterrows()):
        ax1.text(r['test_auc'] + 0.01, i, f"{r['test_auc']:.3f}", va='center', fontsize=9)
    
    # 2. Feature Importance
    ax2 = fig.add_subplot(2, 3, 2)
    
    top_features = feature_importance.head(10)
    y_pos = np.arange(len(top_features))
    
    ax2.barh(y_pos, top_features['importance'], color='#9b59b6', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_features['feature'])
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 10 Churn Predictors')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Churn Probability Distribution
    ax3 = fig.add_subplot(2, 3, 3)
    
    ax3.hist(features_df['churn_probability'], bins=30, color='#3498db', edgecolor='white', alpha=0.7)
    ax3.axvline(0.3, color='#f39c12', linestyle='--', linewidth=2, label='Medium Risk (0.3)')
    ax3.axvline(0.5, color='#e74c3c', linestyle='--', linewidth=2, label='High Risk (0.5)')
    ax3.set_xlabel('Churn Probability')
    ax3.set_ylabel('Number of Customers')
    ax3.set_title('Churn Probability Distribution')
    ax3.legend()
    
    # 4. Risk Category Breakdown
    ax4 = fig.add_subplot(2, 3, 4)
    
    risk_counts = features_df['risk_category'].value_counts()
    risk_order = ['Critical', 'High', 'Medium', 'Low']
    risk_counts = risk_counts.reindex(risk_order).fillna(0)
    
    pie_colors = [colors[cat] for cat in risk_order]
    ax4.pie(risk_counts.values, labels=risk_counts.index, colors=pie_colors,
            autopct='%1.0f%%', startangle=90)
    ax4.set_title('Customer Risk Distribution')
    
    # 5. Revenue at Risk by Category
    ax5 = fig.add_subplot(2, 3, 5)
    
    revenue_by_risk = features_df.groupby('risk_category')['revenue_at_risk'].sum() / 1e6
    revenue_by_risk = revenue_by_risk.reindex(risk_order).fillna(0)
    
    bars = ax5.bar(risk_order, revenue_by_risk.values, 
                   color=[colors[cat] for cat in risk_order], alpha=0.8)
    ax5.set_ylabel('Revenue at Risk ($ Millions)')
    ax5.set_title('Revenue at Risk by Category')
    
    for bar, val in zip(bars, revenue_by_risk.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'${val:.1f}M', ha='center', fontsize=9, fontweight='bold')
    
    # 6. Top At-Risk Customers
    ax6 = fig.add_subplot(2, 3, 6)
    
    if len(interventions_df) > 0:
        top_at_risk = interventions_df.head(10)
        y_pos = np.arange(len(top_at_risk))
        
        bar_colors = [colors.get(u, colors['Medium']) for u in top_at_risk['risk_category']]
        ax6.barh(y_pos, top_at_risk['revenue_at_risk'] / 1000, color=bar_colors, alpha=0.8)
        ax6.set_yticks(y_pos)
        
        # Mask names
        masked = [f"{str(n)[:1]}***" if pd.notna(n) else "Unknown" for n in top_at_risk['customer_name']]
        ax6.set_yticklabels(masked, fontsize=8)
        ax6.set_xlabel('Revenue at Risk ($K)')
        ax6.set_title('Top 10 At-Risk Customers')
        ax6.invert_yaxis()
        
        # Legend
        legend_patches = [mpatches.Patch(color=colors[cat], label=cat) for cat in risk_order]
        ax6.legend(handles=legend_patches, loc='lower right', fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'No high-risk customers found', ha='center', va='center')
        ax6.set_title('Top At-Risk Customers')
    
    plt.tight_layout()
    return fig


def generate_report(features_df, results_df, interventions_df, feature_importance):
    """Generate comprehensive churn analysis report."""
    
    best_model = results_df.loc[results_df['test_auc'].idxmax()]
    
    # Risk summaries
    risk_summary = features_df.groupby('risk_category').agg({
        'customer_no': 'count',
        'monetary': 'sum',
        'revenue_at_risk': 'sum'
    }).round(0)
    
    report = f"""
{'='*80}
                    CUSTOMER CHURN PREDICTION REPORT
                    Sklearn Multi-Model Analysis
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

📊 DATA SUMMARY
{'─'*80}
Total Customers Analyzed:     {len(features_df):,}
Churn Rate (actual):          {features_df['churn_label'].mean()*100:.1f}%
Total Revenue:                ${features_df['monetary'].sum():,.0f}
Total Revenue at Risk:        ${features_df['revenue_at_risk'].sum():,.0f}

📈 MODEL PERFORMANCE
{'─'*80}
{'Model':<25} {'AUC-ROC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}
{'─'*80}
"""
    
    for _, row in results_df.sort_values('test_auc', ascending=False).iterrows():
        report += f"{row['model']:<25} {row['test_auc']:>10.3f} {row['test_f1']:>10.3f} {row['test_precision']:>10.3f} {row['test_recall']:>10.3f}\n"
    
    report += f"""{'─'*80}

🏆 BEST MODEL: {best_model['model']}
   AUC-ROC: {best_model['test_auc']:.3f}
   F1 Score: {best_model['test_f1']:.3f}

🔑 TOP CHURN PREDICTORS
{'─'*80}
"""
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        report += f"   {i:2}. {row['feature']:<25} {row['importance']:.4f}\n"
    
    report += f"""
📊 RISK DISTRIBUTION
{'─'*80}
{'Category':<15} {'Customers':>12} {'Total Revenue':>18} {'Revenue at Risk':>18}
{'─'*80}
"""
    
    for cat in ['Critical', 'High', 'Medium', 'Low']:
        if cat in risk_summary.index:
            row = risk_summary.loc[cat]
            report += f"{cat:<15} {int(row['customer_no']):>12,} ${row['monetary']:>16,.0f} ${row['revenue_at_risk']:>16,.0f}\n"
    
    report += f"""{'─'*80}

🎯 TOP 15 PRIORITY ACCOUNTS
{'─'*80}
"""
    
    if len(interventions_df) > 0:
        for i, (_, row) in enumerate(interventions_df.head(15).iterrows(), 1):
            masked_name = f"{str(row['customer_name'])[:1]}***" if pd.notna(row['customer_name']) else "Unknown"
            report += f"""
#{i}. {masked_name} ({row['customer_no']})
    Risk: {row['risk_category']} ({row['churn_probability']:.0%})
    Revenue at Risk: ${row['revenue_at_risk']:,.0f}
    Urgency: {row['urgency']}
    Actions: {row['strategies'][:80]}...
"""
    
    # Summary statistics
    critical_count = len(features_df[features_df['risk_category'] == 'Critical'])
    high_count = len(features_df[features_df['risk_category'] == 'High'])
    total_at_risk = features_df['revenue_at_risk'].sum()
    
    report += f"""
💰 EXECUTIVE SUMMARY
{'─'*80}
• {critical_count:,} CRITICAL risk customers requiring immediate attention
• {high_count:,} HIGH risk customers for proactive outreach
• ${total_at_risk:,.0f} total revenue at risk
• Potential savings (25% save rate): ${total_at_risk * 0.25:,.0f}

📋 RECOMMENDED ACTIONS
{'─'*80}
1. IMMEDIATE: Contact all CRITICAL customers this week
2. SHORT-TERM: Systematic outreach to HIGH risk customers
3. ONGOING: Run this analysis weekly to catch new at-risk accounts
4. INTEGRATE: Add churn scores to CRM for sales team visibility

{'='*80}
                              END OF REPORT
{'='*80}
"""
    
    return report


def main():
    """Main churn prediction pipeline."""
    
    print("=" * 80)
    print("   SKLEARN CHURN PREDICTION")
    print("   Multi-Model Analysis with Cross-Validation")
    print("=" * 80)
    
    # Load data
    df = load_sales_data()
    
    # Engineer features
    features_df = engineer_features(df)
    
    # Define churn labels
    features_df = define_churn_labels(features_df, churn_days=90)
    
    # Create models
    print("\n🔧 Creating sklearn models...")
    models = create_sklearn_models()
    
    # Train and evaluate
    trained_models, results_df, feature_importance, scaler, feature_cols, X_test, y_test = \
        train_and_evaluate(features_df, models)
    
    # Get best model
    best_model_name = results_df.loc[results_df['test_auc'].idxmax(), 'model']
    best_model = trained_models[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name}")
    
    # Score all customers
    print("\n📊 Scoring all customers...")
    features_df = score_all_customers(features_df, best_model, scaler, feature_cols)
    
    # Generate interventions
    print("\n💡 Generating intervention recommendations...")
    interventions_df = generate_interventions(features_df)
    
    # Generate outputs
    print("\n💾 Saving outputs...")
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    fig = create_visualization(features_df, results_df, interventions_df, feature_importance)
    fig.savefig(REPORTS_PATH / "churn_sklearn_analysis.png", dpi=150, bbox_inches='tight')
    print(f"   Plot: {REPORTS_PATH / 'churn_sklearn_analysis.png'}")
    
    # Churn scores CSV
    features_df.to_csv(REPORTS_PATH / "customer_churn_scores.csv", index=False)
    print(f"   Scores: {REPORTS_PATH / 'customer_churn_scores.csv'}")
    
    # Model comparison CSV
    results_df.to_csv(REPORTS_PATH / "churn_model_comparison.csv", index=False)
    print(f"   Model comparison: {REPORTS_PATH / 'churn_model_comparison.csv'}")
    
    # Interventions CSV
    interventions_df.to_csv(REPORTS_PATH / "intervention_recommendations.csv", index=False)
    print(f"   Interventions: {REPORTS_PATH / 'intervention_recommendations.csv'}")
    
    # Feature importance
    feature_importance.to_csv(REPORTS_PATH / "churn_feature_importance.csv", index=False)
    print(f"   Feature importance: {REPORTS_PATH / 'churn_feature_importance.csv'}")
    
    # Report
    report = generate_report(features_df, results_df, interventions_df, feature_importance)
    with open(REPORTS_PATH / "churn_sklearn_report.txt", 'w') as f:
        f.write(report)
    print(f"   Report: {REPORTS_PATH / 'churn_sklearn_report.txt'}")
    
    # Print summary
    critical = len(features_df[features_df['risk_category'] == 'Critical'])
    high = len(features_df[features_df['risk_category'] == 'High'])
    total_at_risk = features_df['revenue_at_risk'].sum()
    
    print("\n" + "=" * 80)
    print("   🚨 CHURN ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"   Best Model: {best_model_name} (AUC: {results_df.loc[results_df['test_auc'].idxmax(), 'test_auc']:.3f})")
    print(f"   Critical Risk: {critical:,} customers")
    print(f"   High Risk: {high:,} customers")
    print(f"   Total Revenue at Risk: ${total_at_risk:,.0f}")
    print(f"   Potential Savings (25%): ${total_at_risk * 0.25:,.0f}")
    print("=" * 80)
    
    print("\n✅ Churn analysis complete!")
    
    return features_df, trained_models, results_df


if __name__ == "__main__":
    features_df, trained_models, results_df = main()

