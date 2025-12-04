"""
Generate Synthetic Sales Data
=============================
Creates realistic sample data that preserves statistical patterns
(seasonality, trends, distributions) while using completely fake
customer identifiers and anonymized amounts.

This allows sharing the project for educational purposes without
exposing any confidential business data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Fake company name components for realistic-looking names
COMPANY_PREFIXES = [
    "Alpine", "Pacific", "Summit", "Valley", "Mountain", "Coastal", "Metro",
    "Premier", "Elite", "Golden", "Silver", "Crystal", "Blue", "Green", "Red",
    "North", "South", "East", "West", "Central", "United", "National", "Global",
    "First", "Prime", "Superior", "Advanced", "Modern", "Classic", "Heritage"
]

COMPANY_SUFFIXES = [
    "Water", "Beverage", "Distribution", "Supply", "Services", "Solutions",
    "Enterprises", "Industries", "Trading", "Retail", "Wholesale", "Group",
    "Partners", "Associates", "Company", "Corp", "LLC", "Inc", "Systems"
]

COMPANY_TYPES = [
    "Market", "Store", "Shop", "Depot", "Center", "Outlet", "Warehouse",
    "Foods", "Goods", "Products", "Sales", "Direct", "Express", "Plus"
]


def generate_fake_company_name(idx):
    """Generate a realistic-looking fake company name."""
    style = idx % 5
    
    if style == 0:
        # "Alpine Water Distribution"
        return f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
    elif style == 1:
        # "Pacific Water Supply LLC"
        return f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)} {random.choice(['LLC', 'Inc', 'Corp'])}"
    elif style == 2:
        # "Summit Market"
        return f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_TYPES)}"
    elif style == 3:
        # "Valley Foods & Beverage"
        return f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_TYPES)} & {random.choice(COMPANY_SUFFIXES)}"
    else:
        # "Customer_0042" style for variety
        return f"Customer_{idx:04d}"


def generate_customers(n_customers=500):
    """Generate fake customer data."""
    print(f"📋 Generating {n_customers} fake customers...")
    
    customers = []
    for i in range(n_customers):
        cust_id = f"CUST{i:04d}"
        account_no = f"00-{cust_id}"
        name = generate_fake_company_name(i)
        
        customers.append({
            'customer_no': cust_id,
            'account_number': account_no,
            'customer_name': name
        })
    
    return customers


def generate_daily_sales_pattern(start_date, end_date):
    """
    Generate daily sales totals that preserve realistic patterns:
    - Weekly seasonality (Mon-Fri higher, Sat-Sun lower)
    - Some monthly variation
    - Realistic variance
    """
    print("📈 Generating daily sales patterns...")
    
    # Weekly multipliers (based on real data patterns)
    # Mon=1.25, Tue=0.98, Wed=0.99, Thu=1.14, Fri=1.17, Sat=0.14, Sun=0.35
    weekly_pattern = {
        0: 1.25,  # Monday
        1: 0.98,  # Tuesday
        2: 0.99,  # Wednesday
        3: 1.14,  # Thursday
        4: 1.17,  # Friday
        5: 0.14,  # Saturday
        6: 0.35   # Sunday
    }
    
    # Base daily sales (approximate average)
    base_daily_sales = 110000
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_sales = []
    
    for date in dates:
        dow = date.dayofweek
        
        # Apply weekly seasonality
        multiplier = weekly_pattern[dow]
        
        # Add some monthly variation (slight uptick end of month)
        day_of_month = date.day
        if day_of_month > 25:
            multiplier *= 1.1
        elif day_of_month < 5:
            multiplier *= 0.95
        
        # Add yearly trend (slight growth)
        days_from_start = (date - dates[0]).days
        trend_multiplier = 1 + (days_from_start / len(dates)) * 0.15
        
        # Calculate base amount with multipliers
        expected = base_daily_sales * multiplier * trend_multiplier
        
        # Add realistic variance (log-normal for positive skew)
        if dow in [5, 6]:  # Weekend - more variance
            noise = np.random.lognormal(0, 0.5)
        else:
            noise = np.random.lognormal(0, 0.3)
        
        actual = expected * noise
        
        # Occasionally have very low days (holidays, etc.)
        if np.random.random() < 0.02:
            actual *= 0.1
        
        daily_sales.append({
            'date': date,
            'total_sales': max(0, actual),
            'dow': dow
        })
    
    return pd.DataFrame(daily_sales)


def distribute_daily_sales_to_invoices(daily_df, customers):
    """
    Distribute daily sales totals into individual invoices.
    Preserves realistic patterns of invoice counts and sizes.
    """
    print("📝 Distributing sales to individual invoices...")
    
    invoices = []
    invoice_counter = 1000000
    
    # Customer activity levels (some customers order more frequently)
    customer_weights = np.random.exponential(1, len(customers))
    customer_weights = customer_weights / customer_weights.sum()
    
    for _, day in daily_df.iterrows():
        date = day['date']
        total_sales = day['total_sales']
        dow = day['dow']
        
        if total_sales < 100:
            continue
        
        # Determine number of invoices for this day
        if dow in [5, 6]:  # Weekend - fewer invoices
            n_invoices = max(1, int(np.random.poisson(8)))
        else:
            n_invoices = max(1, int(np.random.poisson(35)))
        
        # Generate invoice amounts that sum to daily total
        # Use Dirichlet distribution for realistic spread
        if n_invoices == 1:
            amounts = [total_sales]
        else:
            # Alpha parameter controls spread (lower = more unequal)
            alpha = np.ones(n_invoices) * 0.5
            proportions = np.random.dirichlet(alpha)
            amounts = proportions * total_sales
        
        # Assign to customers
        for amount in amounts:
            # Pick customer weighted by activity level
            cust_idx = np.random.choice(len(customers), p=customer_weights)
            customer = customers[cust_idx]
            
            invoices.append({
                'invoice_number': f"INV-{invoice_counter:07d}",
                'invoice_date': date.strftime('%Y-%m-%d'),
                'customer_no': customer['customer_no'],
                'account_number': customer['account_number'],
                'customer_name': customer['customer_name'],
                'amount': round(amount, 2),
                'invoice_type': 'IN',
                'source': 'Sample'
            })
            
            invoice_counter += 1
    
    return invoices


def generate_sample_data():
    """Main function to generate complete sample dataset."""
    
    print("=" * 60)
    print("   GENERATING SYNTHETIC SALES DATA")
    print("=" * 60)
    
    # Configuration
    start_date = '2023-01-01'
    end_date = '2025-12-04'
    n_customers = 500
    
    # Step 1: Generate fake customers
    customers = generate_customers(n_customers)
    
    # Step 2: Generate daily sales pattern
    daily_df = generate_daily_sales_pattern(start_date, end_date)
    
    print(f"   Generated {len(daily_df)} days of data")
    print(f"   Total sales: ${daily_df['total_sales'].sum():,.0f}")
    print(f"   Daily average: ${daily_df['total_sales'].mean():,.0f}")
    
    # Step 3: Distribute to invoices
    invoices = distribute_daily_sales_to_invoices(daily_df, customers)
    
    print(f"   Generated {len(invoices):,} invoices")
    
    # Step 4: Save to JSON
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PATH / "combined_sales_history.json"
    
    with open(output_path, 'w') as f:
        json.dump(invoices, f, indent=2)
    
    print(f"\n💾 Saved to {output_path}")
    
    # Summary statistics
    df = pd.DataFrame(invoices)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['amount'] = pd.to_numeric(df['amount'])
    
    print("\n" + "=" * 60)
    print("   📊 SAMPLE DATA SUMMARY")
    print("=" * 60)
    print(f"   Total invoices: {len(df):,}")
    print(f"   Unique customers: {df['customer_no'].nunique()}")
    print(f"   Date range: {df['invoice_date'].min().date()} to {df['invoice_date'].max().date()}")
    print(f"   Total sales: ${df['amount'].sum():,.0f}")
    print(f"   Average invoice: ${df['amount'].mean():,.2f}")
    
    # Weekly pattern verification
    daily = df.groupby('invoice_date')['amount'].sum().reset_index()
    daily['dow'] = pd.to_datetime(daily['invoice_date']).dt.dayofweek
    weekly = daily.groupby('dow')['amount'].mean()
    
    print("\n   Weekly Pattern (should show Mon-Fri high, Sat-Sun low):")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, d in enumerate(days):
        print(f"   {d}: ${weekly.get(i, 0):,.0f}")
    
    print("\n✅ Sample data generation complete!")
    
    return invoices


if __name__ == "__main__":
    generate_sample_data()

