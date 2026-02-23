"""
================================================================================
E-Commerce Order Lifecycle Funnel Analysis
================================================================================
Purpose: Define order funnel stages and create binary indicators
Author: Data Analyst
Date: 2026-01-20

Funnel Stages:
1. Order Placed (created, approved)
2. Payment Approved (approved)
3. Shipped (shipped)
4. Delivered (delivered)
================================================================================
"""

import pandas as pd
import numpy as np
import sys

# Output to both console and file
class OutputWriter:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

output = OutputWriter('funnel_analysis_results.txt')
sys.stdout = output

print("=" * 70)
print("ORDER LIFECYCLE FUNNEL ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD MERGED DATASET
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOAD MERGED DATASET")
print("=" * 70)

df = pd.read_csv('merged_ecommerce_data.csv')
print(f"\nLoaded dataset: {len(df):,} rows x {df.shape[1]} columns")

# =============================================================================
# STEP 2: EXPLORE ORDER STATUS VALUES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: EXPLORE ORDER STATUS VALUES")
print("=" * 70)

# Get unique order_id level data (avoid counting duplicates from items)
orders_unique = df.drop_duplicates(subset='order_id')
total_orders = len(orders_unique)

print(f"\nTotal Unique Orders: {total_orders:,}")

status_counts = orders_unique['order_status'].value_counts()
print("\nOrder Status Distribution:")
print("-" * 50)
for status, count in status_counts.items():
    pct = count / total_orders * 100
    print(f"  {status:<20} : {count:>7,} orders ({pct:>5.2f}%)")

# =============================================================================
# STEP 3: DEFINE FUNNEL STAGES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: DEFINE FUNNEL STAGES")
print("=" * 70)

print("""
ORDER LIFECYCLE FUNNEL DEFINITION:
----------------------------------

Stage 1: ORDER PLACED
  - Includes: All orders that entered the system
  - Status: created, approved, invoiced, processing, shipped, delivered
  - EXCLUDES: canceled, unavailable
  
Stage 2: PAYMENT APPROVED  
  - Includes: Orders where payment was successfully processed
  - Status: approved, invoiced, processing, shipped, delivered
  - EXCLUDES: created (still pending), canceled, unavailable

Stage 3: SHIPPED
  - Includes: Orders handed off to shipping carrier
  - Status: shipped, delivered
  - EXCLUDES: created, approved, processing, canceled, unavailable

Stage 4: DELIVERED
  - Includes: Orders successfully received by customer
  - Status: delivered only
  - EXCLUDES: All other statuses
""")

# Define which statuses belong to each funnel stage
funnel_definitions = {
    'stage_1_placed': ['created', 'approved', 'invoiced', 'processing', 'shipped', 'delivered'],
    'stage_2_approved': ['approved', 'invoiced', 'processing', 'shipped', 'delivered'],
    'stage_3_shipped': ['shipped', 'delivered'],
    'stage_4_delivered': ['delivered']
}

# Excluded statuses
excluded_statuses = ['canceled', 'unavailable']

# =============================================================================
# STEP 4: CREATE BINARY INDICATOR COLUMNS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: CREATE BINARY INDICATOR COLUMNS")
print("=" * 70)

# Create binary indicators for each stage
df['funnel_order_placed'] = df['order_status'].isin(funnel_definitions['stage_1_placed']).astype(int)
df['funnel_payment_approved'] = df['order_status'].isin(funnel_definitions['stage_2_approved']).astype(int)
df['funnel_shipped'] = df['order_status'].isin(funnel_definitions['stage_3_shipped']).astype(int)
df['funnel_delivered'] = df['order_status'].isin(funnel_definitions['stage_4_delivered']).astype(int)

# Also create an excluded indicator
df['funnel_excluded'] = df['order_status'].isin(excluded_statuses).astype(int)

print("\nNew Binary Columns Created:")
print("  1. funnel_order_placed    (1 = order entered system)")
print("  2. funnel_payment_approved (1 = payment successful)")
print("  3. funnel_shipped          (1 = handed to carrier)")
print("  4. funnel_delivered        (1 = received by customer)")
print("  5. funnel_excluded         (1 = canceled or unavailable)")

# Verify column creation with sample
print("\nSample of new columns (unique orders):")
sample_cols = ['order_id', 'order_status', 'funnel_order_placed', 
               'funnel_payment_approved', 'funnel_shipped', 'funnel_delivered', 'funnel_excluded']
sample = df.drop_duplicates('order_id')[sample_cols].head(10)
print(sample.to_string(index=False))

# =============================================================================
# STEP 5: CALCULATE FUNNEL METRICS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: CALCULATE FUNNEL METRICS")
print("=" * 70)

# Calculate at order level (not row level)
orders_df = df.drop_duplicates('order_id')

# Funnel counts
stage_1_count = orders_df['funnel_order_placed'].sum()
stage_2_count = orders_df['funnel_payment_approved'].sum()
stage_3_count = orders_df['funnel_shipped'].sum()
stage_4_count = orders_df['funnel_delivered'].sum()
excluded_count = orders_df['funnel_excluded'].sum()

print("\nFUNNEL COUNTS (Unique Orders):")
print("-" * 50)
print(f"  Stage 1 - Order Placed:     {stage_1_count:>7,} orders")
print(f"  Stage 2 - Payment Approved: {stage_2_count:>7,} orders")
print(f"  Stage 3 - Shipped:          {stage_3_count:>7,} orders")
print(f"  Stage 4 - Delivered:        {stage_4_count:>7,} orders")
print(f"  EXCLUDED:                   {excluded_count:>7,} orders")

# Conversion rates
print("\nCONVERSION RATES:")
print("-" * 50)
print(f"  Placed → Approved:  {stage_2_count/stage_1_count*100:>6.2f}%")
print(f"  Approved → Shipped: {stage_3_count/stage_2_count*100:>6.2f}%")
print(f"  Shipped → Delivered:{stage_4_count/stage_3_count*100:>6.2f}%")
print(f"  Overall (Placed → Delivered): {stage_4_count/stage_1_count*100:>6.2f}%")

# Drop-off rates
print("\nDROP-OFF RATES (Where customers are lost):")
print("-" * 50)
dropoff_1_2 = stage_1_count - stage_2_count
dropoff_2_3 = stage_2_count - stage_3_count
dropoff_3_4 = stage_3_count - stage_4_count

print(f"  After Placed (not approved):  {dropoff_1_2:>6,} orders ({dropoff_1_2/stage_1_count*100:.2f}%)")
print(f"  After Approved (not shipped): {dropoff_2_3:>6,} orders ({dropoff_2_3/stage_2_count*100:.2f}%)")
print(f"  After Shipped (not delivered):{dropoff_3_4:>6,} orders ({dropoff_3_4/stage_3_count*100:.2f}%)")

# =============================================================================
# STEP 6: EXPLAIN EXCLUDED ORDERS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: WHY EXCLUDE CANCELED & UNAVAILABLE ORDERS?")
print("=" * 70)

# Get breakdown of excluded orders
excluded_orders = orders_df[orders_df['funnel_excluded'] == 1]
excluded_breakdown = excluded_orders['order_status'].value_counts()

print("\nExcluded Orders Breakdown:")
for status, count in excluded_breakdown.items():
    pct = count / total_orders * 100
    print(f"  {status}: {count:,} orders ({pct:.2f}%)")

print("""
BUSINESS RATIONALE FOR EXCLUSION:
----------------------------------

1. CANCELED ORDERS
   - These are orders that were intentionally stopped by customer or seller
   - They did NOT progress through the natural order lifecycle
   - Including them would SKEW our conversion rates downward
   - They require SEPARATE analysis (cancellation reasons, timing, etc.)

2. UNAVAILABLE ORDERS  
   - Product was not available after order was placed
   - This is a SUPPLY-SIDE issue, not a customer journey issue
   - Should be analyzed separately for inventory management

3. WHY SEPARATE ANALYSIS?
   - Main funnel measures: "How well do we fulfill customer intent?"
   - Canceled/unavailable represent: "Why did fulfillment fail?"
   - Different questions = Different analyses

4. BUSINESS IMPACT
   - Funnel without exclusions: Shows true conversion capability
   - Canceled analysis: Identifies customer satisfaction issues
   - Unavailable analysis: Identifies inventory/supply chain issues

RECOMMENDATION: Analyze excluded orders separately with root cause analysis.
""")

# =============================================================================
# STEP 7: SUMMARY STATISTICS BY STATUS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: ORDER STATUS SUMMARY")
print("=" * 70)

print("\nStatus → Funnel Stage Mapping:")
print("-" * 70)
print(f"{'Status':<15} {'Count':>10} {'% Total':>10} {'In Funnel?':>12} {'Reached Stage':>15}")
print("-" * 70)

for status in orders_df['order_status'].unique():
    count = (orders_df['order_status'] == status).sum()
    pct = count / total_orders * 100
    in_funnel = 'Yes' if status not in excluded_statuses else 'EXCLUDED'
    
    if status == 'delivered':
        stage = 'Stage 4'
    elif status == 'shipped':
        stage = 'Stage 3'
    elif status in ['approved', 'invoiced', 'processing']:
        stage = 'Stage 2'
    elif status == 'created':
        stage = 'Stage 1'
    else:
        stage = 'N/A'
    
    print(f"{status:<15} {count:>10,} {pct:>9.2f}% {in_funnel:>12} {stage:>15}")

# =============================================================================
# STEP 8: SAVE UPDATED DATASET
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: SAVE UPDATED DATASET")
print("=" * 70)

# Save the dataset with funnel indicators
df.to_csv('merged_ecommerce_with_funnel.csv', index=False)
print(f"\nSaved updated dataset: merged_ecommerce_with_funnel.csv")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {df.shape[1]} (added 5 funnel indicator columns)")

print("\nNew columns added:")
new_cols = [col for col in df.columns if col.startswith('funnel_')]
for col in new_cols:
    print(f"  - {col}")

# =============================================================================
# STEP 9: VISUAL FUNNEL SUMMARY (ASCII)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: VISUAL FUNNEL SUMMARY")
print("=" * 70)

print("""
                    ORDER LIFECYCLE FUNNEL
                    ======================
                    
    ┌─────────────────────────────────────────────────────┐
    │                 ORDER PLACED                        │
    │                  {placed:,} orders                       │
    │                    (100%)                           │
    └─────────────────────────────────────────────────────┘
                           │
                           ▼ {conv1:.1f}% converted
    ┌───────────────────────────────────────────────┐
    │              PAYMENT APPROVED                 │
    │               {approved:,} orders                  │
    └───────────────────────────────────────────────┘
                           │
                           ▼ {conv2:.1f}% converted  
    ┌─────────────────────────────────────────┐
    │               SHIPPED                   │
    │            {shipped:,} orders               │
    └─────────────────────────────────────────┘
                           │
                           ▼ {conv3:.1f}% converted
    ┌───────────────────────────────────┐
    │           DELIVERED               │
    │         {delivered:,} orders          │
    │       ({overall:.1f}% of placed)       │
    └───────────────────────────────────┘
    
    
    ════════════════════════════════════════════
    EXCLUDED FROM FUNNEL: {excluded:,} orders
    (Canceled: {canceled:,} | Unavailable: {unavail:,})
    ════════════════════════════════════════════
""".format(
    placed=stage_1_count,
    approved=stage_2_count,
    shipped=stage_3_count,
    delivered=stage_4_count,
    conv1=stage_2_count/stage_1_count*100,
    conv2=stage_3_count/stage_2_count*100,
    conv3=stage_4_count/stage_3_count*100,
    overall=stage_4_count/stage_1_count*100,
    excluded=excluded_count,
    canceled=excluded_breakdown.get('canceled', 0),
    unavail=excluded_breakdown.get('unavailable', 0)
))

print("\n" + "=" * 70)
print("FUNNEL ANALYSIS COMPLETE")
print("=" * 70)

# Close output
output.close()
sys.stdout = output.terminal

print("\nResults saved to: funnel_analysis_results.txt")
print("Updated dataset saved to: merged_ecommerce_with_funnel.csv")
