#!/usr/bin/env python3
"""
Run All Forecasting Models
==========================
Convenience script to run the entire forecasting pipeline.

Usage:
    python run_all.py
"""

import subprocess
import sys
from pathlib import Path

# Scripts to run in order
SCRIPTS = [
    ("Prophet (Tuned)", "src/prophet_tuned.py"),
    ("SARIMAX", "src/sarimax_forecast.py"),
    ("XGBoost", "src/xgboost_forecast.py"),
    ("Scikit-Learn", "src/sklearn_forecast.py"),
    ("Ensemble", "src/ensemble_forecast.py"),
    ("Model Validation", "src/model_validation.py"),
    ("Churn Prediction", "src/churn_sklearn.py"),
]

def main():
    print("=" * 60)
    print("   SALES FORECASTING - FULL PIPELINE")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    failed = []
    
    for name, script in SCRIPTS:
        print(f"\n{'─' * 60}")
        print(f"▶ Running: {name}")
        print(f"{'─' * 60}")
        
        script_path = project_root / script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root)
        )
        
        if result.returncode != 0:
            print(f"❌ {name} failed!")
            failed.append(name)
        else:
            print(f"✅ {name} complete!")
    
    print("\n" + "=" * 60)
    if failed:
        print(f"   ⚠️  COMPLETED WITH ERRORS: {', '.join(failed)}")
    else:
        print("   ✅ ALL MODELS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nCheck the reports/ folder for all outputs.")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())

