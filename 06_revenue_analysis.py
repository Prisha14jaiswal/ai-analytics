"""
================================================================================
E-Commerce Revenue Analysis
================================================================================
Purpose: Analyze revenue metrics - total, AOV, by state
Author: Data Analyst
Date: 2026-01-20
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Output writer
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

output = OutputWriter('revenue_analysis_results.txt')
sys.stdout = output

print("=" * 70)
print("E-COMMERCE REVENUE ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOAD DATA")
print("=" * 70)

df = pd.read_csv('merged_ecommerce_with_funnel.csv')
print(f"\nLoaded dataset: {len(df):,} rows")

# =============================================================================
# STEP 2: CALCULATE TOTAL REVENUE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: TOTAL REVENUE METRICS")
print("=" * 70)

# Revenue = price + freight_value (for delivered items - but we'll show both)
total_product_revenue = df['price'].sum()
total_freight_revenue = df['freight_value'].sum()
total_revenue = total_product_revenue + total_freight_revenue

print(f"""
TOTAL REVENUE BREAKDOWN:
------------------------
  Product Revenue:  ${total_product_revenue:>15,.2f}
  Freight Revenue:  ${total_freight_revenue:>15,.2f}
  ─────────────────────────────────────
  TOTAL REVENUE:    ${total_revenue:>15,.2f}

Revenue Split:
  - Product: {total_product_revenue/total_revenue*100:.1f}%
  - Freight: {total_freight_revenue/total_revenue*100:.1f}%
""")

# =============================================================================
# STEP 3: CALCULATE AVERAGE ORDER VALUE (AOV)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: AVERAGE ORDER VALUE (AOV)")
print("=" * 70)

# Calculate order-level metrics (sum all items in an order)
order_revenue = df.groupby('order_id').agg({
    'price': 'sum',
    'freight_value': 'sum',
    'order_item_id': 'count'
}).reset_index()

order_revenue.columns = ['order_id', 'order_product_value', 'order_freight_value', 'items_count']
order_revenue['order_total_value'] = order_revenue['order_product_value'] + order_revenue['order_freight_value']

# Calculate AOV
total_orders = len(order_revenue)
aov_product = order_revenue['order_product_value'].mean()
aov_freight = order_revenue['order_freight_value'].mean()
aov_total = order_revenue['order_total_value'].mean()

# Additional order value metrics
median_order_value = order_revenue['order_total_value'].median()
min_order_value = order_revenue['order_total_value'].min()
max_order_value = order_revenue['order_total_value'].max()
std_order_value = order_revenue['order_total_value'].std()

print(f"""
AVERAGE ORDER VALUE (AOV):
--------------------------
  Total Orders:           {total_orders:,}
  
  AOV (Product only):     ${aov_product:,.2f}
  AOV (Freight only):     ${aov_freight:,.2f}
  AOV (Total):            ${aov_total:,.2f}
  
ORDER VALUE DISTRIBUTION:
  Minimum:                ${min_order_value:,.2f}
  Median:                 ${median_order_value:,.2f}
  Maximum:                ${max_order_value:,.2f}
  Std Dev:                ${std_order_value:,.2f}
  
  Average items per order: {order_revenue['items_count'].mean():.2f}
""")

# Order value percentiles
percentiles = [10, 25, 50, 75, 90, 95, 99]
print("ORDER VALUE PERCENTILES:")
for p in percentiles:
    val = order_revenue['order_total_value'].quantile(p/100)
    print(f"  {p}th percentile: ${val:,.2f}")

# =============================================================================
# STEP 4: REVENUE BY STATE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: REVENUE BY STATE")
print("=" * 70)

# Merge state info to order revenue
df_state = df[['order_id', 'customer_state']].drop_duplicates()
order_revenue = order_revenue.merge(df_state, on='order_id', how='left')

# Aggregate by state
state_revenue = order_revenue.groupby('customer_state').agg({
    'order_id': 'count',
    'order_product_value': 'sum',
    'order_freight_value': 'sum',
    'order_total_value': ['sum', 'mean']
}).reset_index()

state_revenue.columns = ['state', 'order_count', 'product_revenue', 'freight_revenue', 'total_revenue', 'aov']
state_revenue['revenue_share'] = (state_revenue['total_revenue'] / state_revenue['total_revenue'].sum() * 100).round(2)
state_revenue = state_revenue.sort_values('total_revenue', ascending=False)

print("\nREVENUE BY STATE (Top 15):")
print("-" * 100)
print(f"{'State':<8} {'Orders':>10} {'Product Rev':>15} {'Freight Rev':>15} {'Total Rev':>15} {'Share':>8} {'AOV':>10}")
print("-" * 100)

for _, row in state_revenue.head(15).iterrows():
    print(f"{row['state']:<8} {row['order_count']:>10,} ${row['product_revenue']:>14,.0f} ${row['freight_revenue']:>14,.0f} ${row['total_revenue']:>14,.0f} {row['revenue_share']:>7.1f}% ${row['aov']:>9,.2f}")

print("-" * 100)

# Summary stats
print(f"\nTotal States: {len(state_revenue)}")
print(f"Top 3 states account for: {state_revenue.head(3)['revenue_share'].sum():.1f}% of revenue")
print(f"Top 5 states account for: {state_revenue.head(5)['revenue_share'].sum():.1f}% of revenue")

# =============================================================================
# STEP 5: AOV COMPARISON BY STATE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: AOV COMPARISON BY STATE")
print("=" * 70)

# Filter to states with 500+ orders for meaningful comparison
significant_states = state_revenue[state_revenue['order_count'] >= 500].copy()

top_aov = significant_states.nlargest(5, 'aov')
bottom_aov = significant_states.nsmallest(5, 'aov')

print("\nTOP 5 STATES BY AOV (min 500 orders):")
print("-" * 50)
for i, (_, row) in enumerate(top_aov.iterrows(), 1):
    print(f"  {i}. {row['state']}: ${row['aov']:,.2f} AOV ({row['order_count']:,} orders)")

print("\nBOTTOM 5 STATES BY AOV (min 500 orders):")
print("-" * 50)
for i, (_, row) in enumerate(bottom_aov.iterrows(), 1):
    print(f"  {i}. {row['state']}: ${row['aov']:,.2f} AOV ({row['order_count']:,} orders)")

# Calculate AOV difference
aov_gap = top_aov.iloc[0]['aov'] - bottom_aov.iloc[0]['aov']
print(f"\nAOV Gap (highest vs lowest): ${aov_gap:,.2f}")

# =============================================================================
# STEP 6: CREATE VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: CREATE VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('E-Commerce Revenue Analysis', fontsize=16, fontweight='bold')

# --- Chart 1: Top 10 States by Revenue ---
ax1 = axes[0, 0]
top_10_revenue = state_revenue.nlargest(10, 'total_revenue')
colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, 10))[::-1]
bars1 = ax1.barh(top_10_revenue['state'], top_10_revenue['total_revenue'] / 1000000, color=colors1)

for bar, rev_share in zip(bars1, top_10_revenue['revenue_share']):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
             f'{rev_share:.1f}%', va='center', fontsize=9)

ax1.set_xlabel('Revenue (Millions $)', fontsize=11)
ax1.set_title('Top 10 States by Total Revenue', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Chart 2: Order Value Distribution ---
ax2 = axes[0, 1]
# Cap at 99th percentile for visualization
cap_value = order_revenue['order_total_value'].quantile(0.99)
order_values_capped = order_revenue['order_total_value'][order_revenue['order_total_value'] <= cap_value]

ax2.hist(order_values_capped, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
ax2.axvline(x=aov_total, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: ${aov_total:.0f}')
ax2.axvline(x=median_order_value, color='#27ae60', linestyle='--', linewidth=2, label=f'Median: ${median_order_value:.0f}')

ax2.set_xlabel('Order Value ($)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Distribution of Order Values (up to 99th percentile)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Chart 3: AOV by State (Top 10 by volume) ---
ax3 = axes[1, 0]
top_10_volume = state_revenue.nlargest(10, 'order_count')
top_10_volume_sorted = top_10_volume.sort_values('aov', ascending=True)

overall_aov = aov_total
colors3 = ['#27ae60' if x >= overall_aov else '#e74c3c' for x in top_10_volume_sorted['aov']]
bars3 = ax3.barh(top_10_volume_sorted['state'], top_10_volume_sorted['aov'], color=colors3)

ax3.axvline(x=overall_aov, color='#3498db', linestyle='--', linewidth=2, label=f'Overall AOV: ${overall_aov:.0f}')

for bar in bars3:
    ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
             f'${bar.get_width():.0f}', va='center', fontsize=9)

ax3.set_xlabel('Average Order Value ($)', fontsize=11)
ax3.set_title('AOV by State (Top 10 by Order Volume)', fontsize=13, fontweight='bold')
ax3.legend(loc='lower right')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# --- Chart 4: Revenue Composition (Product vs Freight) ---
ax4 = axes[1, 1]
top_10_rev = state_revenue.nlargest(10, 'total_revenue')

x = np.arange(len(top_10_rev))
width = 0.7

bars_product = ax4.bar(x, top_10_rev['product_revenue'] / 1000000, width, label='Product Revenue', color='#3498db')
bars_freight = ax4.bar(x, top_10_rev['freight_revenue'] / 1000000, width, 
                       bottom=top_10_rev['product_revenue'] / 1000000, label='Freight Revenue', color='#f39c12')

ax4.set_xticks(x)
ax4.set_xticklabels(top_10_rev['state'], fontsize=10)
ax4.set_ylabel('Revenue (Millions $)', fontsize=11)
ax4.set_xlabel('State', fontsize=11)
ax4.set_title('Revenue Composition: Product vs Freight (Top 10 States)', fontsize=13, fontweight='bold')
ax4.legend()
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('revenue_analysis_charts.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\nVisualization saved: revenue_analysis_charts.png")

# =============================================================================
# STEP 7: EXPLAIN REVENUE DIFFERENCES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: WHAT DRIVES REVENUE DIFFERENCES ACROSS REGIONS")
print("=" * 70)

# Calculate some metrics for explanation
sp_data = state_revenue[state_revenue['state'] == 'SP'].iloc[0]
rj_data = state_revenue[state_revenue['state'] == 'RJ'].iloc[0]

print(f"""
FACTORS DRIVING REGIONAL REVENUE DIFFERENCES
=============================================

1. POPULATION & MARKET SIZE
   ---------------------------
   - SP (São Paulo): {sp_data['revenue_share']:.1f}% of total revenue
     → Brazil's largest state by population & economy
     → Major urban center with high consumer spending
   
   - Top 5 states = {state_revenue.head(5)['revenue_share'].sum():.1f}% of revenue
     → Revenue concentrated in economically developed regions

2. AVERAGE ORDER VALUE (AOV) VARIATIONS
   -------------------------------------
   - Highest AOV: ${top_aov.iloc[0]['aov']:.2f} ({top_aov.iloc[0]['state']})
   - Lowest AOV:  ${bottom_aov.iloc[0]['aov']:.2f} ({bottom_aov.iloc[0]['state']})
   - Gap: ${aov_gap:.2f} ({aov_gap/bottom_aov.iloc[0]['aov']*100:.1f}% difference)
   
   WHY AOV DIFFERS:
   - Income levels: Wealthier regions buy higher-priced items
   - Product preferences: Some regions favor premium products
   - Freight costs: Remote areas pay more for shipping

3. FREIGHT IMPACT BY REGION
   -------------------------
   - Freight is ~{total_freight_revenue/total_revenue*100:.1f}% of total revenue
   - Remote states pay higher freight (distance from distribution centers)
   - This can both increase revenue AND reduce conversions

4. ORDER FREQUENCY & CUSTOMER BASE
   --------------------------------
   - SP: {sp_data['order_count']:,} orders ({sp_data['order_count']/total_orders*100:.1f}% of all orders)
   - High-population states have larger customer base
   - More urban = more e-commerce adoption

5. ECONOMIC FACTORS
   -----------------
   - Southeast region (SP, RJ, MG, ES): Most developed economy
   - Northeast region: Lower average income, lower AOV
   - This explains both volume AND value differences

KEY BUSINESS INSIGHTS:
======================

1. CONCENTRATION RISK
   - Heavy reliance on SP ({sp_data['revenue_share']:.1f}% of revenue)
   - Diversifying into other regions could reduce risk

2. GROWTH OPPORTUNITIES  
   - States with high AOV but low volume = underserved markets
   - States with high volume but low AOV = upselling opportunity

3. PRICING STRATEGY
   - Consider region-specific pricing for freight
   - Premium products may sell better in high-income regions

4. MARKET EXPANSION
   - Northeast states have growth potential (large population)
   - Investment in logistics could unlock these markets
""")

# =============================================================================
# STEP 8: SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: EXECUTIVE SUMMARY")
print("=" * 70)

print(f"""
REVENUE ANALYSIS SUMMARY
========================

TOTAL REVENUE: ${total_revenue:,.2f}
  - Product: ${total_product_revenue:,.2f} ({total_product_revenue/total_revenue*100:.1f}%)
  - Freight: ${total_freight_revenue:,.2f} ({total_freight_revenue/total_revenue*100:.1f}%)

AVERAGE ORDER VALUE: ${aov_total:,.2f}
  - Median Order Value: ${median_order_value:,.2f}

TOP 3 STATES BY REVENUE:
  1. {state_revenue.iloc[0]['state']}: ${state_revenue.iloc[0]['total_revenue']:,.0f} ({state_revenue.iloc[0]['revenue_share']:.1f}%)
  2. {state_revenue.iloc[1]['state']}: ${state_revenue.iloc[1]['total_revenue']:,.0f} ({state_revenue.iloc[1]['revenue_share']:.1f}%)
  3. {state_revenue.iloc[2]['state']}: ${state_revenue.iloc[2]['total_revenue']:,.0f} ({state_revenue.iloc[2]['revenue_share']:.1f}%)

TOP 3 STATES BY AOV (500+ orders):
  1. {top_aov.iloc[0]['state']}: ${top_aov.iloc[0]['aov']:,.2f}
  2. {top_aov.iloc[1]['state']}: ${top_aov.iloc[1]['aov']:,.2f}
  3. {top_aov.iloc[2]['state']}: ${top_aov.iloc[2]['aov']:,.2f}
""")

print("\n" + "=" * 70)
print("REVENUE ANALYSIS COMPLETE")
print("=" * 70)

# Close output
output.close()
sys.stdout = output.terminal

print("\nResults saved to: revenue_analysis_results.txt")
print("Visualization saved to: revenue_analysis_charts.png")
